import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip
import pickle
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize Indonesian stemmer and stopword remover
factory = StemmerFactory()
stemmer = factory.create_stemmer()

factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()

# Load model
@st.cache_resource
def load_model():
    try:
        with gzip.open('random_forest_final_model.sav.gz', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load training data for TF-IDF vectorizer
@st.cache_data
def load_training_data():
    try:
        df = pd.read_csv('inacoved-deployment.csv')
        return df
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return None

# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords
    text = stopword_remover.remove(text)
    
    # Stemming
    text = stemmer.stem(text)
    
    return text

# Create TF-IDF vectorizer from training data
@st.cache_resource
def create_tfidf_vectorizer():
    df = load_training_data()
    if df is not None and 'stemming_data' in df.columns:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        vectorizer.fit(df['stemming_data'].fillna(''))
        return vectorizer
    return None

# Predict sentiment
def predict_sentiment(text, model, vectorizer):
    if not text.strip():
        return "Netral", 0.0
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Vectorize text
    text_vector = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    # Map prediction to label
    if prediction == 0:
        sentiment = "Negatif"
        confidence = probability[0]
    elif prediction == 1:
        sentiment = "Netral"
        confidence = probability[1]
    else:
        sentiment = "Positif"
        confidence = probability[2]
    
    return sentiment, confidence

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Analisis Sentimen Indonesia",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Analisis Sentimen Bahasa Indonesia")
    st.markdown("---")
    
    # Load model and vectorizer
    model = load_model()
    vectorizer = create_tfidf_vectorizer()
    
    if model is None or vectorizer is None:
        st.error("Gagal memuat model atau data training. Pastikan file model dan data tersedia.")
        return
    
    # Sidebar
    st.sidebar.header("📊 Informasi Model")
    st.sidebar.info("""
    **Model**: Random Forest Classifier
    **Dataset**: InaCOVEd
    **Kategori Sentimen**:
    - 🔴 Negatif
    - 🟡 Netral  
    - 🟢 Positif
    """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Input Teks")
        
        # Text input methods
        input_method = st.radio(
            "Pilih metode input:",
            ["Ketik teks", "Upload file CSV"]
        )
        
        if input_method == "Ketik teks":
            # Single text input
            user_input = st.text_area(
                "Masukkan teks yang ingin dianalisis:",
                height=150,
                placeholder="Contoh: Saya sangat senang dengan pelayanan ini..."
            )
            
            if st.button("🔍 Analisis Sentimen", type="primary"):
                if user_input.strip():
                    with st.spinner("Menganalisis sentimen..."):
                        sentiment, confidence = predict_sentiment(user_input, model, vectorizer)
                        
                        # Display results
                        st.markdown("### 📊 Hasil Analisis")
                        
                        # Sentiment with emoji
                        emoji_map = {"Positif": "🟢", "Netral": "🟡", "Negatif": "🔴"}
                        st.markdown(f"**Sentimen**: {emoji_map[sentiment]} **{sentiment}**")
                        st.markdown(f"**Confidence**: {confidence:.2%}")
                        
                        # Progress bar for confidence
                        st.progress(confidence)
                        
                        # Processed text
                        processed = preprocess_text(user_input)
                        with st.expander("Lihat teks yang telah diproses"):
                            st.text(processed)
                
                else:
                    st.warning("Silakan masukkan teks terlebih dahulu!")
        
        else:
            # Batch processing
            uploaded_file = st.file_uploader(
                "Upload file CSV dengan kolom 'text':",
                type=['csv']
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    if 'text' not in df.columns:
                        st.error("File CSV harus memiliki kolom 'text'")
                    else:
                        st.write("Preview data:")
                        st.dataframe(df.head())
                        
                        if st.button("🔍 Analisis Batch", type="primary"):
                            with st.spinner("Menganalisis sentimen untuk semua teks..."):
                                sentiments = []
                                confidences = []
                                
                                progress_bar = st.progress(0)
                                for i, text in enumerate(df['text']):
                                    sentiment, confidence = predict_sentiment(str(text), model, vectorizer)
                                    sentiments.append(sentiment)
                                    confidences.append(confidence)
                                    progress_bar.progress((i + 1) / len(df))
                                
                                # Add results to dataframe
                                df['sentiment'] = sentiments
                                df['confidence'] = confidences
                                
                                # Display results
                                st.markdown("### 📊 Hasil Analisis Batch")
                                st.dataframe(df)
                                
                                # Download results
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Hasil",
                                    data=csv,
                                    file_name="sentiment_analysis_results.csv",
                                    mime="text/csv"
                                )
                                
                                # Batch statistics
                                st.markdown("### 📈 Statistik Hasil")
                                sentiment_counts = df['sentiment'].value_counts()
                                
                                # Pie chart
                                fig = px.pie(
                                    values=sentiment_counts.values,
                                    names=sentiment_counts.index,
                                    title="Distribusi Sentimen",
                                    color_discrete_map={
                                        'Positif': '#00ff00',
                                        'Netral': '#ffff00',
                                        'Negatif': '#ff0000'
                                    }
                                )
                                st.plotly_chart(fig)
                
                except Exception as e:
                    st.error(f"Error membaca file: {e}")
    
    with col2:
        st.header("ℹ️ Informasi")
        
        # Model info
        with st.expander("📋 Detail Model"):
            st.write("""
            **Random Forest Classifier**
            - Ensemble learning method
            - Robust terhadap overfitting
            - Memberikan feature importance
            - Cocok untuk klasifikasi teks
            """)
        
        # Preprocessing info
        with st.expander("🔧 Preprocessing"):
            st.write("""
            **Tahapan Preprocessing**:
            1. Lowercase conversion
            2. Penghapusan karakter khusus
            3. Stopword removal
            4. Stemming (Sastrawi)
            5. TF-IDF Vectorization
            """)
        
        # Tips
        with st.expander("💡 Tips Penggunaan"):
            st.write("""
            **Untuk hasil terbaik**:
            - Gunakan teks dalam Bahasa Indonesia
            - Teks minimal 3-5 kata
            - Hindari teks yang terlalu pendek
            - Konteks yang jelas akan memberikan hasil lebih akurat
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Aplikasi Analisis Sentimen Bahasa Indonesia menggunakan Random Forest"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()