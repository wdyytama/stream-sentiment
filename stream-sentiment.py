import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen Indonesia",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk preprocessing teks
@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

@st.cache_resource
def load_models():
    """Load TF-IDF vectorizer dan model Random Forest"""
    try:
        # Load TF-IDF vectorizer
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load Random Forest model
        with gzip.open('model random forest.pkl.gz', 'rb') as f:
            model = pickle.load(f)
        
        return vectorizer, model
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def clean_text(text):
    """Membersihkan teks dari karakter tidak diinginkan"""
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Hapus mention dan hashtag
    text = re.sub(r'@\w+|#\w+', '', text)
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    # Hapus karakter khusus, hanya simpan huruf dan spasi
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def preprocess_text(text, stemmer):
    """Preprocessing lengkap untuk teks"""
    # Clean text
    text = clean_text(text)
    
    # Stemming
    text = stemmer.stem(text)
    
    return text

def predict_sentiment(text, vectorizer, model, stemmer):
    """Prediksi sentimen dari teks"""
    # Preprocessing
    processed_text = preprocess_text(text, stemmer)
    
    # Vectorization
    text_vector = vectorizer.transform([processed_text])
    
    # Prediction
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    # Map prediction to label
    label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
    sentiment_label = label_map[prediction]
    
    # Get confidence score
    confidence = max(probability) * 100
    
    return sentiment_label, confidence, probability

# Load data untuk analisis batch
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('inacoved-deployment.csv')
        return df
    except FileNotFoundError:
        st.warning("File inacoved-deployment.csv tidak ditemukan. Fitur analisis batch tidak tersedia.")
        return None

# Main App
def main():
    st.markdown('<h1 class="main-header">🎭 Analisis Sentimen Bahasa Indonesia</h1>', unsafe_allow_html=True)
    
    # Load models
    vectorizer, model = load_models()
    if vectorizer is None or model is None:
        st.error("Gagal memuat model. Pastikan file tfidf_vectorizer.pkl dan model random forest.pkl.gz tersedia.")
        return
    
    stemmer = load_stemmer()
    
    # Sidebar
    st.sidebar.title("🎯 Menu Navigasi")
    page = st.sidebar.selectbox(
        "Pilih Halaman:",
        ["Prediksi Tunggal", "Analisis Batch", "Dashboard Analytics", "Tentang Model"]
    )
    
    if page == "Prediksi Tunggal":
        st.header("📝 Prediksi Sentimen Tunggal")
        
        # Input teks
        user_input = st.text_area(
            "Masukkan teks untuk dianalisis:",
            placeholder="Contoh: Pelayanan di rumah sakit ini sangat baik dan ramah",
            height=100
        )
        
        if st.button("🔍 Analisis Sentimen", type="primary"):
            if user_input.strip():
                with st.spinner("Menganalisis sentimen..."):
                    sentiment, confidence, probabilities = predict_sentiment(
                        user_input, vectorizer, model, stemmer
                    )
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if sentiment == "Positif":
                        st.markdown(f"""
                        <div class="sentiment-positive">
                            <h3>✅ Sentimen: {sentiment}</h3>
                            <p>Tingkat Kepercayaan: {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif sentiment == "Negatif":
                        st.markdown(f"""
                        <div class="sentiment-negative">
                            <h3>❌ Sentimen: {sentiment}</h3>
                            <p>Tingkat Kepercayaan: {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="sentiment-neutral">
                            <h3>⚖️ Sentimen: {sentiment}</h3>
                            <p>Tingkat Kepercayaan: {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Probability chart
                    labels = ['Negatif', 'Netral', 'Positif']
                    fig = go.Figure(data=[
                        go.Bar(x=labels, y=probabilities*100, 
                               marker_color=['#dc3545', '#ffc107', '#28a745'])
                    ])
                    fig.update_layout(
                        title="Distribusi Probabilitas",
                        yaxis_title="Probabilitas (%)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Mohon masukkan teks untuk dianalisis.")
    
    elif page == "Analisis Batch":
        st.header("📊 Analisis Batch")
        
        uploaded_file = st.file_uploader(
            "Upload file CSV dengan kolom 'text':",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                
                if 'text' not in df_upload.columns:
                    st.error("File harus memiliki kolom 'text'")
                else:
                    st.write("Preview data:")
                    st.dataframe(df_upload.head())
                    
                    if st.button("🚀 Jalankan Analisis Batch"):
                        with st.spinner("Menganalisis seluruh data..."):
                            results = []
                            progress_bar = st.progress(0)
                            
                            for i, text in enumerate(df_upload['text']):
                                if pd.notna(text):
                                    sentiment, confidence, _ = predict_sentiment(
                                        str(text), vectorizer, model, stemmer
                                    )
                                    results.append({
                                        'text': text,
                                        'sentiment': sentiment,
                                        'confidence': confidence
                                    })
                                progress_bar.progress((i + 1) / len(df_upload))
                            
                            results_df = pd.DataFrame(results)
                            
                            # Display results
                            st.success("Analisis selesai!")
                            st.dataframe(results_df)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Hasil",
                                data=csv,
                                file_name='hasil_analisis_sentimen.csv',
                                mime='text/csv'
                            )
                            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    elif page == "Dashboard Analytics":
        st.header("📈 Dashboard Analytics")
        
        # Load data
        df = load_data()
        if df is not None:
            # Visualisasi distribusi sentimen
            sentiment_counts = df['sentiment'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Distribusi Sentimen dalam Dataset",
                    color_discrete_map={
                        'Positif': '#28a745',
                        'Negatif': '#dc3545',
                        'Netral': '#ffc107'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Bar chart
                fig_bar = px.bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    title="Jumlah Data per Sentimen",
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'Positif': '#28a745',
                        'Negatif': '#dc3545',
                        'Netral': '#ffc107'
                    }
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Statistik
            st.subheader("📊 Statistik Dataset")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Data", len(df))
            with col2:
                st.metric("Data Positif", sentiment_counts.get('Positif', 0))
            with col3:
                st.metric("Data Negatif", sentiment_counts.get('Negatif', 0))
            with col4:
                st.metric("Data Netral", sentiment_counts.get('Netral', 0))
        else:
            st.info("Data tidak tersedia untuk dashboard analytics.")
    
    elif page == "Tentang Model":
        st.header("🤖 Tentang Model")
        
        st.markdown("""
        ### Model Analisis Sentimen
        
        Model ini menggunakan **Random Forest Classifier** dengan preprocessing teks bahasa Indonesia yang meliputi:
        
        #### Preprocessing:
        - **Text Cleaning**: Menghapus URL, mention, hashtag, angka, dan karakter khusus
        - **Case Normalization**: Mengubah semua teks menjadi lowercase
        - **Stemming**: Menggunakan Sastrawi untuk stemming bahasa Indonesia
        
        #### Feature Extraction:
        - **TF-IDF Vectorization**: Mengkonversi teks menjadi vektor numerik
        
        #### Model:
        - **Random Forest**: Ensemble learning dengan multiple decision trees
        - **Classes**: 3 kelas sentimen (Positif, Negatif, Netral)
        
        #### Performa Model:
        Model telah dilatih dan dievaluasi menggunakan dataset yang telah dipreprocessing untuk memberikan prediksi sentimen yang akurat pada teks bahasa Indonesia.
        """)
        
        st.subheader("🔧 Cara Menggunakan")
        st.markdown("""
        1. **Prediksi Tunggal**: Masukkan satu teks untuk dianalisis sentimennya
        2. **Analisis Batch**: Upload file CSV dengan kolom 'text' untuk analisis massal
        3. **Dashboard Analytics**: Lihat visualisasi distribusi sentimen dalam dataset
        """)

if __name__ == "__main__":
    main()
