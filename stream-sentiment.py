import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

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
    .probability-bar {
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
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
        
        # Debug info
        st.sidebar.write("🔧 Model Info:")
        st.sidebar.write(f"Vectorizer features: {len(vectorizer.get_feature_names_out())}")
        st.sidebar.write(f"Model expects: {model.n_features_in_} features")
        
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

def predict_sentiment_dummy(text):
    """Prediksi sentimen dummy untuk testing"""
    import random
    
    # Simple rule-based prediction for demo
    text_lower = text.lower()
    
    positive_words = ['baik', 'bagus', 'senang', 'suka', 'hebat', 'mantap', 'keren']
    negative_words = ['buruk', 'jelek', 'sedih', 'benci', 'kecewa', 'marah', 'parah']
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "Positif", random.uniform(70, 95), [0.1, 0.2, 0.7]
    elif neg_count > pos_count:
        return "Negatif", random.uniform(70, 95), [0.7, 0.2, 0.1]
    else:
        return "Netral", random.uniform(60, 80), [0.3, 0.4, 0.3]
    """Prediksi sentimen dari teks"""
    try:
        # Preprocessing
        processed_text = preprocess_text(text, stemmer)
        
        # Debug: tampilkan teks yang diprocess
        st.sidebar.write("🔍 Debug Info:")
        st.sidebar.write(f"Original: {text[:50]}...")
        st.sidebar.write(f"Processed: {processed_text[:50]}...")
        
        # Vectorization
        text_vector = vectorizer.transform([processed_text])
        
        # Debug: tampilkan dimensi vector
        st.sidebar.write(f"Vector shape: {text_vector.shape}")
        st.sidebar.write(f"Vector features: {text_vector.shape[1]}")
        
        # Prediction
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        
        # Map prediction to label
        label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
        sentiment_label = label_map[prediction]
        
        # Get confidence score
        confidence = max(probability) * 100
        
        return sentiment_label, confidence, probability
        
    except ValueError as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        st.error("Kemungkinan model dan vectorizer tidak kompatibel")
        return "Error", 0, [0, 0, 0]
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return "Error", 0, [0, 0, 0]

def display_probability_bars(probabilities):
    """Menampilkan probability bars menggunakan HTML/CSS"""
    labels = ['Negatif', 'Netral', 'Positif']
    colors = ['#dc3545', '#ffc107', '#28a745']
    
    st.subheader("Distribusi Probabilitas")
    
    for i, (label, prob, color) in enumerate(zip(labels, probabilities, colors)):
        percentage = prob * 100
        st.markdown(f"**{label}**: {percentage:.2f}%")
        
        # Create progress bar
        st.progress(prob)
        
        # Alternative: HTML bar
        bar_html = f"""
        <div style="background-color: #f0f0f0; border-radius: 10px; padding: 3px; margin: 5px 0;">
            <div style="background-color: {color}; width: {percentage}%; height: 20px; border-radius: 7px; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;">
                {percentage:.1f}%
            </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

# Load data untuk analisis batch
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('inacoved-deployment.csv')
        return df
    except FileNotFoundError:
        st.warning("File inacoved-deployment.csv tidak ditemukan. Fitur analisis batch tidak tersedia.")
        return None

def create_simple_chart(sentiment_counts):
    """Membuat chart sederhana menggunakan streamlit native"""
    st.subheader("Distribusi Sentimen")
    
    # Menggunakan bar_chart bawaan streamlit
    chart_data = pd.DataFrame({
        'Sentimen': sentiment_counts.index,
        'Jumlah': sentiment_counts.values
    }).set_index('Sentimen')
    
    st.bar_chart(chart_data)
    
    # Tambahan: menampilkan dalam bentuk metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="😠 Negatif",
            value=sentiment_counts.get('Negatif', 0),
            delta=f"{(sentiment_counts.get('Negatif', 0)/sentiment_counts.sum()*100):.1f}%"
        )
    
    with col2:
        st.metric(
            label="😐 Netral", 
            value=sentiment_counts.get('Netral', 0),
            delta=f"{(sentiment_counts.get('Netral', 0)/sentiment_counts.sum()*100):.1f}%"
        )
    
    with col3:
        st.metric(
            label="😊 Positif",
            value=sentiment_counts.get('Positif', 0),
            delta=f"{(sentiment_counts.get('Positif', 0)/sentiment_counts.sum()*100):.1f}%"
        )

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
                    # Display probability menggunakan fungsi custom
                    display_probability_bars(probabilities)
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
                            
                            # Show simple analytics
                            sentiment_dist = results_df['sentiment'].value_counts()
                            create_simple_chart(sentiment_dist)
                            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    elif page == "Dashboard Analytics":
        st.header("📈 Dashboard Analytics")
        
        # Load data
        df = load_data()
        if df is not None:
            # Visualisasi distribusi sentimen menggunakan streamlit native
            sentiment_counts = df['sentiment'].value_counts()
            
            create_simple_chart(sentiment_counts)
            
            # Statistik detail
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
                
            # Tampilkan sample data
            st.subheader("📋 Sample Data")
            st.dataframe(df.head(10))
            
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
        
        st.subheader("📚 Dependencies")
        st.code("""
streamlit==1.28.0
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
nltk==3.8.1
Sastrawi==1.0.1
joblib==1.3.2
        """)

if __name__ == "__main__":
    main()
