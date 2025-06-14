import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import re
import os
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
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #28a745;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #dc3545;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #ffc107;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk preprocessing teks
@st.cache_resource
def load_stemmer():
    """Load Sastrawi stemmer"""
    try:
        factory = StemmerFactory()
        return factory.create_stemmer()
    except Exception as e:
        st.error(f"Error loading stemmer: {e}")
        return None

@st.cache_resource
def load_models():
    """Load TF-IDF vectorizer dan model Random Forest"""
    try:
        # Cek apakah file ada
        if not os.path.exists('tfidf_vectorizer.pkl'):
            st.error("File tfidf_vectorizer.pkl tidak ditemukan!")
            return None, None
            
        if not os.path.exists('rf_sentiment_model.pkl.gz') and not os.path.exists('rf_sentiment_model.pkl'):
            st.error("File model tidak ditemukan!")
            return None, None
        
        # Load TF-IDF vectorizer
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load Random Forest model (coba .gz dulu, kalau tidak ada coba .pkl)
        try:
            with gzip.open('rf_sentiment_model.pkl.gz', 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError:
            with open('rf_sentiment_model.pkl', 'rb') as f:
                model = pickle.load(f)
        
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def clean_text(text):
    """Membersihkan teks dari karakter tidak diinginkan"""
    if not isinstance(text, str):
        text = str(text)
        
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
    if stemmer is None:
        return clean_text(text)
        
    # Clean text
    text = clean_text(text)
    
    # Stemming
    try:
        text = stemmer.stem(text)
    except Exception as e:
        st.warning(f"Stemming error: {e}")
        # Return cleaned text without stemming if stemming fails
        pass
    
    return text

def predict_sentiment(text, vectorizer, model, stemmer):
    """Prediksi sentimen dari teks"""
    try:
        # Preprocessing
        processed_text = preprocess_text(text, stemmer)
        
        if not processed_text.strip():
            return "Netral", 50.0, [0.33, 0.34, 0.33]
        
        # Vectorization
        text_vector = vectorizer.transform([processed_text])
        
        # Prediction
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        
        # Map prediction to label
        label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
        sentiment_label = label_map.get(prediction, 'Netral')
        
        # Get confidence score
        confidence = max(probability) * 100
        
        return sentiment_label, confidence, probability
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return "Netral", 0.0, [0.33, 0.34, 0.33]

# Load data untuk analisis batch
@st.cache_data
def load_data():
    try:
        if os.path.exists('inacoved-deployment.csv'):
            df = pd.read_csv('inacoved-deployment.csv')
            return df
        else:
            st.warning("File inacoved-deployment.csv tidak ditemukan. Fitur analisis batch tidak tersedia.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_sample_data():
    """Buat data sample untuk demo jika file tidak ada"""
    sample_data = {
        'sentiment': ['Positif'] * 150 + ['Negatif'] * 100 + ['Netral'] * 75
    }
    return pd.DataFrame(sample_data)

# Main App
def main():
    st.markdown('<h1 class="main-header">🎭 Analisis Sentimen Bahasa Indonesia</h1>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models..."):
        vectorizer, model = load_models()
        stemmer = load_stemmer()
    
    if vectorizer is None or model is None:
        st.error("❌ Gagal memuat model. Pastikan file tfidf_vectorizer.pkl dan rf_sentiment_model.pkl/.gz tersedia.")
        st.info("💡 Upload file model Anda atau hubungi administrator.")
        return
    
    if stemmer is None:
        st.warning("⚠️ Stemmer tidak dapat dimuat. Analisis akan dilakukan tanpa stemming.")
    
    # Sidebar
    st.sidebar.title("🎯 Menu Navigasi")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Pilih Halaman:",
        ["🏠 Beranda", "📝 Prediksi Tunggal", "📊 Analisis Batch", "📈 Dashboard Analytics", "🤖 Tentang Model"]
    )
    
    if page == "🏠 Beranda":
        st.header("🏠 Selamat Datang di Analisis Sentimen Indonesia")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📝 Prediksi Tunggal</h3>
                <p>Analisis sentimen untuk satu teks</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Analisis Batch</h3>
                <p>Analisis massal dari file CSV</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📈 Dashboard</h3>
                <p>Visualisasi data dan statistik</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        ### 🚀 Fitur Unggulan:
        - **Preprocessing Lengkap**: Text cleaning dan stemming bahasa Indonesia
        - **Model Canggih**: Random Forest dengan TF-IDF vectorization
        - **Visualisasi Interaktif**: Dashboard dengan chart dan grafik
        - **Batch Processing**: Analisis ribuan teks sekaligus
        - **Real-time**: Prediksi instan dengan confidence score
        """)
    
    elif page == "📝 Prediksi Tunggal":
        st.header("📝 Prediksi Sentimen Tunggal")
        
        # Input teks
        user_input = st.text_area(
            "Masukkan teks untuk dianalisis:",
            placeholder="Contoh: Pelayanan di rumah sakit ini sangat baik dan ramah sekali, dokternya profesional",
            height=120,
            help="Masukkan teks dalam bahasa Indonesia untuk dianalisis sentimennya"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.button("🔍 Analisis Sentimen", type="primary")
        
        if analyze_button:
            if user_input.strip():
                with st.spinner("🔄 Menganalisis sentimen..."):
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
                            <p><strong>Tingkat Kepercayaan: {confidence:.2f}%</strong></p>
                            <p>Teks menunjukkan sentimen yang positif dan optimis.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif sentiment == "Negatif":
                        st.markdown(f"""
                        <div class="sentiment-negative">
                            <h3>❌ Sentimen: {sentiment}</h3>
                            <p><strong>Tingkat Kepercayaan: {confidence:.2f}%</strong></p>
                            <p>Teks menunjukkan sentimen yang negatif atau kritik.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="sentiment-neutral">
                            <h3>⚖️ Sentimen: {sentiment}</h3>
                            <p><strong>Tingkat Kepercayaan: {confidence:.2f}%</strong></p>
                            <p>Teks menunjukkan sentimen yang netral atau informatif.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Probability chart
                    labels = ['Negatif', 'Netral', 'Positif']
                    colors = ['#dc3545', '#ffc107', '#28a745']
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=labels, 
                            y=probabilities*100, 
                            marker_color=colors,
                            text=[f'{p*100:.1f}%' for p in probabilities],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="Distribusi Probabilitas",
                        yaxis_title="Probabilitas (%)",
                        height=350,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Processed text info
                with st.expander("ℹ️ Detail Preprocessing"):
                    processed = preprocess_text(user_input, stemmer)
                    st.write("**Teks Original:**")
                    st.write(user_input)
                    st.write("**Teks Setelah Preprocessing:**")
                    st.write(processed)
                    
            else:
                st.warning("⚠️ Mohon masukkan teks untuk dianalisis.")
    
    elif page == "📊 Analisis Batch":
        st.header("📊 Analisis Batch")
        
        st.markdown("Upload file CSV dengan kolom **'text'** untuk analisis sentimen massal.")
        
        uploaded_file = st.file_uploader(
            "Pilih file CSV:",
            type=['csv'],
            help="File harus memiliki kolom 'text' yang berisi teks untuk dianalisis"
        )
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                
                if 'text' not in df_upload.columns:
                    st.error("❌ File harus memiliki kolom 'text'")
                    st.info("Format yang benar:")
                    st.code("text\nContoh teks pertama\nContoh teks kedua")
                else:
                    st.success(f"✅ File berhasil diupload! Ditemukan {len(df_upload)} baris data.")
                    
                    with st.expander("👁️ Preview Data"):
                        st.dataframe(df_upload.head(10))
                    
                    if st.button("🚀 Jalankan Analisis Batch", type="primary"):
                        with st.spinner("🔄 Menganalisis seluruh data..."):
                            results = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, text in enumerate(df_upload['text']):
                                if pd.notna(text) and str(text).strip():
                                    sentiment, confidence, _ = predict_sentiment(
                                        str(text), vectorizer, model, stemmer
                                    )
                                    results.append({
                                        'text': text,
                                        'sentiment': sentiment,
                                        'confidence': confidence
                                    })
                                else:
                                    results.append({
                                        'text': text,
                                        'sentiment': 'Netral',
                                        'confidence': 0.0
                                    })
                                
                                progress = (i + 1) / len(df_upload)
                                progress_bar.progress(progress)
                                status_text.text(f"Progress: {i+1}/{len(df_upload)} ({progress*100:.1f}%)")
                            
                            results_df = pd.DataFrame(results)
                            
                            # Display results
                            st.success("🎉 Analisis selesai!")
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            sentiment_counts = results_df['sentiment'].value_counts()
                            
                            with col1:
                                st.metric("Total Data", len(results_df))
                            with col2:
                                st.metric("Positif", sentiment_counts.get('Positif', 0))
                            with col3:
                                st.metric("Negatif", sentiment_counts.get('Negatif', 0))
                            with col4:
                                st.metric("Netral", sentiment_counts.get('Netral', 0))
                            
                            # Results table
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Hasil Analisis",
                                data=csv,
                                file_name='hasil_analisis_sentimen.csv',
                                mime='text/csv',
                                type="primary"
                            )
                            
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                st.info("Pastikan file CSV valid dan memiliki kolom 'text'")
    
    elif page == "📈 Dashboard Analytics":
        st.header("📈 Dashboard Analytics")
        
        # Load data
        df = load_data()
        if df is None:
            st.info("Data tidak tersedia. Menggunakan data sample untuk demo.")
            df = create_sample_data()
        
        if df is not None and 'sentiment' in df.columns:
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
                    },
                    hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
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
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Statistik detail
            st.subheader("📊 Statistik Dataset")
            col1, col2, col3, col4 = st.columns(4)
            
            total_data = len(df)
            positif_count = sentiment_counts.get('Positif', 0)
            negatif_count = sentiment_counts.get('Negatif', 0)
            netral_count = sentiment_counts.get('Netral', 0)
            
            with col1:
                st.metric("Total Data", total_data)
            with col2:
                st.metric("Data Positif", positif_count, f"{positif_count/total_data*100:.1f}%")
            with col3:
                st.metric("Data Negatif", negatif_count, f"{negatif_count/total_data*100:.1f}%")
            with col4:
                st.metric("Data Netral", netral_count, f"{netral_count/total_data*100:.1f}%")
            
            # Persentase dalam bentuk progress bar
            st.subheader("📈 Distribusi Persentase")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Positif**")
                st.progress(positif_count/total_data)
                st.write(f"{positif_count/total_data*100:.1f}%")
            
            with col2:
                st.write("**Negatif**")
                st.progress(negatif_count/total_data)
                st.write(f"{negatif_count/total_data*100:.1f}%")
            
            with col3:
                st.write("**Netral**")
                st.progress(netral_count/total_data)
                st.write(f"{netral_count/total_data*100:.1f}%")
                
        else:
            st.error("❌ Data tidak valid atau kolom 'sentiment' tidak ditemukan.")
    
    elif page == "🤖 Tentang Model":
        st.header("🤖 Tentang Model")
        
        tab1, tab2, tab3 = st.tabs(["📋 Overview", "🔧 Technical Details", "📚 Cara Penggunaan"])
        
        with tab1:
            st.markdown("""
            ### 🎯 Analisis Sentimen Bahasa Indonesia
            
            Aplikasi ini menggunakan **Random Forest Classifier** yang telah dilatih khusus untuk menganalisis sentimen teks bahasa Indonesia. Model dapat mengklasifikasikan teks ke dalam tiga kategori:
            
            - ✅ **Positif**: Sentimen yang menunjukkan perasaan baik, puas, atau optimis
            - ❌ **Negatif**: Sentimen yang menunjukkan ketidakpuasan, kritik, atau pesimis  
            - ⚖️ **Netral**: Sentimen yang objektif, informatif, atau tidak memihak
            
            ### 🎯 Keunggulan Model:
            - **Akurasi Tinggi**: Dilatih dengan dataset yang besar dan seimbang
            - **Preprocessing Lengkap**: Text cleaning dan stemming bahasa Indonesia
            - **Fast Prediction**: Prediksi real-time dengan confidence score
            - **Robust**: Dapat menangani berbagai jenis teks informal dan formal
            """)
        
        with tab2:
            st.markdown("""
            ### 🔍 Detail Teknis
            
            #### Preprocessing Pipeline:
            ```
            1. Text Cleaning
               ├── Hapus URL, mention (@), hashtag (#)
               ├── Hapus angka dan karakter khusus
               ├── Konversi ke lowercase
               └── Normalisasi spasi
            
            2. Stemming (Sastrawi)
               └── Mengubah kata ke bentuk dasar
            
            3. Feature Extraction (TF-IDF)
               ├── Max features: 5000
               ├── N-gram range: (1,2)
               └── Vectorization
            ```
            
            #### Model Architecture:
            - **Algorithm**: Random Forest Classifier
            - **Data Balancing**: SMOTE (Synthetic Minority Oversampling)
            - **Feature Selection**: TF-IDF dengan bigram
            - **Cross Validation**: Stratified split
            
            #### Model Performance:
            - **Training**: Menggunakan dataset berlabel dengan distribusi seimbang
            - **Validation**: Stratified train-test split (80:20)
            - **Metrics**: Accuracy, Precision, Recall, F1-Score
            """)
        
        with tab3:
            st.markdown("""
            ### 📚 Panduan Penggunaan
            
            #### 1. 📝 Prediksi Tunggal
            - Masukkan teks yang ingin dianalisis
            - Klik tombol "Analisis Sentimen"
            - Lihat hasil prediksi dan confidence score
            - Periksa distribusi probabilitas untuk setiap kelas
            
            #### 2. 📊 Analisis Batch
            - Siapkan file CSV dengan kolom 'text'
            - Upload file melalui file uploader
            - Klik "Jalankan Analisis Batch"
            - Download hasil dalam format CSV
            
            #### 3. 📈 Dashboard Analytics
            - Lihat visualisasi distribusi sentimen
            - Analisis statistik dataset
            - Monitor persentase setiap kategori sentimen
            
            #### 💡 Tips untuk Hasil Terbaik:
            - Gunakan teks dalam bahasa Indonesia
            - Hindari teks yang terlalu pendek (< 3 kata)
            - Teks dengan konteks yang jelas memberikan hasil lebih akurat
            - Untuk analisis batch, pastikan data bersih dari nilai kosong
            """)
        
        st.markdown("---")
        st.info("🚀 **Model ini dikembangkan untuk membantu analisis sentimen teks bahasa Indonesia dengan akurasi dan kecepatan tinggi.**")

if __name__ == "__main__":
    main()
