import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Set page config for better UI
st.set_page_config(
    page_title="Emotion Detection & Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download stopwords once, using Streamlit's caching
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load emotion model and vectorizer once
@st.cache_resource
def load_emotion_model_and_vectorizer():
    with open('models/emotion_detection_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('models/emotion_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Load sentiment model and vectorizer once
@st.cache_resource
def load_sentiment_model_and_vectorizer():
    with open('models/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('models/vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Define emotion prediction function
def predict_emotion(text, model, vectorizer, stop_words):
    # Preprocess text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)
    
    # Predict emotion
    emotion = model.predict(text)
    return emotion[0]

# Define sentiment prediction function
def predict_sentiment(text, model, vectorizer, stop_words):
    # Preprocess text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)
    
    # Predict sentiment
    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"



# Emotion colors mapping - Bright & Fun theme
EMOTION_CONFIG = {
    'joy': {'color': '#fff8e1', 'border': '#ffd54f', 'text': '#f57f17', 'badge_bg': '#fffde7'},
    'sad': {'color': '#e1f5fe', 'border': '#29b6f6', 'text': '#0277bd', 'badge_bg': '#b3e5fc'},
    'anger': {'color': '#ffebee', 'border': '#ef5350', 'text': '#c62828', 'badge_bg': '#ffcdd2'},
    'fear': {'color': '#f3e5f5', 'border': '#ab47bc', 'text': '#6a1b9a', 'badge_bg': '#e1bee7'},
    'love': {'color': '#fce4ec', 'border': '#ec407a', 'text': '#ad1457', 'badge_bg': '#f8bbd0'},
    'surprise': {'color': '#ffe0b2', 'border': '#ff9800', 'text': '#e65100', 'badge_bg': '#ffe0b2'}
}

# Function to create a colored card
def create_card(tweet_text, emotion, sentiment):
    emotion_config = EMOTION_CONFIG.get(emotion.lower(), EMOTION_CONFIG['joy'])
    sentiment_color = "#c8e6c9" if sentiment == "Positive" else "#ffcccc"
    sentiment_border = "#4caf50" if sentiment == "Positive" else "#f44336"
    sentiment_text = "#1b5e20" if sentiment == "Positive" else "#b71c1c"
    
    card_html = f"""
    <div style="
        background: {emotion_config['color']};
        border-left: 6px solid {emotion_config['border']};
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; gap: 16px;">
            <div style="flex: 1;">
                <div style="
                    color: {emotion_config['text']};
                    font-size: 12px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-bottom: 6px;
                ">Detected Emotion</div>
                <div style="
                    color: {emotion_config['text']};
                    font-size: 32px;
                    font-weight: 800;
                ">{emotion.capitalize()}</div>
            </div>
            <div style="
                background: {sentiment_color};
                border: 2px solid {sentiment_border};
                padding: 12px 20px;
                border-radius: 10px;
                font-size: 14px;
                font-weight: 700;
                color: {sentiment_text};
                white-space: nowrap;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            ">
                {sentiment}
            </div>
        </div>
        <div style="
            border-top: 2px solid {emotion_config['border']};
            padding-top: 16px;
            margin-top: 16px;
        ">
            <p style="color: #2d3748; margin: 0; font-size: 15px; line-height: 1.7; font-weight: 500;">{tweet_text}</p>
        </div>
    </div>
    """
    return card_html

# Main app logic
def main():
    # Custom CSS - Bright & Fun UI
    st.markdown("""
    <style>
        * {
            font-family: 'Poppins', 'Segoe UI', sans-serif;
        }
        
        body {
            background-color: #ffffff;
            color: #2d3748;
        }
        
        .main {
            background-color: #ffffff;
            border-radius: 20px;
            padding: 20px;
        }
        
        .main-title {
            text-align: center;
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 6px;
            color: #667eea;
            letter-spacing: -1px;
        }
        
        .subtitle {
            text-align: center;
            color: #764ba2;
            margin-bottom: 16px;
            font-size: 14px;
            font-weight: 500;
            letter-spacing: 0.5px;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 3px solid rgba(102, 126, 234, 0.2);
            background: #f5f7ff;
            padding: 6px;
            border-radius: 12px;
            margin-bottom: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            background: #ffffff;
            border-radius: 10px;
            font-weight: 600;
            color: #764ba2;
            border: 2px solid transparent;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
            box-shadow: 0 2px 6px rgba(102, 126, 234, 0.06);
            font-size: 14px;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #667eea;
            color: white;
            border: 2px solid #667eea;
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
            transform: translateY(-1px);
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: #667eea;
            border: 2px solid #667eea;
            transform: translateY(-1px);
            box-shadow: 0 2px 12px rgba(102, 126, 234, 0.15);
        }
        
        .stSubheader {
            color: #667eea;
            font-weight: 700;
            font-size: 20px;
            margin-bottom: 12px;
        }
        
        .stButton > button {
            background: #667eea;
            color: white;
            font-weight: 700;
            border: none;
            border-radius: 10px;
            padding: 10px 24px;
            font-size: 14px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #667eea;
            background-color: #f8faff;
            color: #2d3748;
            font-size: 14px;
            padding: 12px;
            transition: all 0.3s ease;
        }
        
        .stTextArea textarea:focus {
            border-color: #764ba2;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
            background-color: #ffffff;
        }
        
        .stTextInput input {
            border-radius: 12px;
            border: 2px solid #667eea;
            background-color: #f8faff;
            color: #2d3748;
            font-size: 16px;
            padding: 12px 16px;
            transition: all 0.3s ease;
        }
        
        .stTextInput input:focus {
            border-color: #764ba2;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            background-color: #ffffff;
        }
        
        .stSuccess {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            color: #0d5e2a;
            border: 2px solid #2ecc71;
            border-radius: 12px;
            padding: 16px;
            font-weight: 600;
            box-shadow: 0 8px 20px rgba(46, 204, 113, 0.2);
        }
        
        .stWarning {
            background: linear-gradient(135deg, #ffd89b 0%, #ffe5b4 100%);
            color: #7a4e0d;
            border: 2px solid #f39c12;
            border-radius: 12px;
            padding: 16px;
            font-weight: 600;
            box-shadow: 0 8px 20px rgba(243, 156, 18, 0.2);
        }
        
        .stError {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            color: #c0392b;
            border: 2px solid #e74c3c;
            border-radius: 12px;
            padding: 16px;
            font-weight: 600;
            box-shadow: 0 8px 20px rgba(231, 76, 60, 0.2);
        }
        
        .stInfo {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            color: #0e3d62;
            border: 2px solid #3498db;
            border-radius: 12px;
            padding: 16px;
            font-weight: 600;
            box-shadow: 0 8px 20px rgba(52, 152, 219, 0.2);
        }
        
        [data-testid="stDivider"] {
            background: #667eea;
            border: none;
            height: 2px;
            margin: 12px 0;
            border-radius: 2px;
        }
        
        .stCaption {
            color: #764ba2;
            font-size: 12px;
            text-align: center;
            font-weight: 600;
            margin-top: 12px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-title">Emotion Detection & Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Detect emotions and analyze sentiment in text</div>', unsafe_allow_html=True)
    
    st.divider()

    # Load stopwords, models, vectorizers only once
    stop_words = load_stopwords()
    emotion_model, emotion_vectorizer = load_emotion_model_and_vectorizer()
    sentiment_model, sentiment_vectorizer = load_sentiment_model_and_vectorizer()

    # Input Section
    st.markdown('<div style="margin-bottom: 8px;"><h3 style="color: #667eea; margin: 0; font-size: 18px;">Enter Text</h3></div>', unsafe_allow_html=True)
    text_input = st.text_area(
        "Text to analyze",
        placeholder="Type or paste your text here...",
        height=100,
        label_visibility="collapsed"
    )
    
    # Analyze Button
    col1, col2, col3 = st.columns([1, 4, 4])
    with col1:
        analyze_button = st.button("Analyze", use_container_width=True, key="analyze_btn")
    
    # Results Section
    if analyze_button or text_input.strip():
        if text_input.strip():
            if analyze_button:
                emotion = predict_emotion(text_input, emotion_model, emotion_vectorizer, stop_words)
                sentiment = predict_sentiment(text_input, sentiment_model, sentiment_vectorizer, stop_words)
                st.session_state.last_emotion = emotion
                st.session_state.last_sentiment = sentiment
                st.session_state.last_text = text_input
            elif hasattr(st.session_state, 'last_emotion'):
                emotion = st.session_state.last_emotion
                sentiment = st.session_state.last_sentiment
                text_input = st.session_state.last_text
            else:
                emotion = predict_emotion(text_input, emotion_model, emotion_vectorizer, stop_words)
                sentiment = predict_sentiment(text_input, sentiment_model, sentiment_vectorizer, stop_words)
            
            st.markdown('<div style="margin-top: 12px; margin-bottom: 8px;"><h3 style="color: #667eea; margin: 0; font-size: 18px;">Results</h3></div>', unsafe_allow_html=True)
            
            # Results - Emotion and Sentiment stacked vertically
            # Emotion Card
            emotion_config = EMOTION_CONFIG.get(emotion.lower(), EMOTION_CONFIG['joy'])
            emotion_card = f"""
            <div style="
                background: {emotion_config['color']};
                border-left: 6px solid {emotion_config['border']};
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                min-height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            ">
                <div style="
                    color: {emotion_config['text']};
                    font-size: 11px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-bottom: 6px;
                ">Emotion</div>
                <div style="
                    color: {emotion_config['text']};
                    font-size: 32px;
                    font-weight: 800;
                ">{emotion.capitalize()}</div>
            </div>
            """
            st.markdown(emotion_card, unsafe_allow_html=True)
            
            # Sentiment Card
            sentiment_color = "#c8e6c9" if sentiment == "Positive" else "#ffcccc"
            sentiment_border = "#4caf50" if sentiment == "Positive" else "#f44336"
            sentiment_text = "#1b5e20" if sentiment == "Positive" else "#b71c1c"
            sentiment_display = "Positive" if sentiment == "Positive" else "Negative"
            
            sentiment_card = f"""
            <div style="
                background: {sentiment_color};
                border-left: 6px solid {sentiment_border};
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                min-height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            ">
                <div style="
                    color: {sentiment_text};
                    font-size: 11px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-bottom: 6px;
                ">Sentiment</div>
                <div style="
                    color: {sentiment_text};
                    font-size: 32px;
                    font-weight: 800;
                ">{sentiment_display}</div>
            </div>
            """
            st.markdown(sentiment_card, unsafe_allow_html=True)
            
            # Text Display Card
            text_card = f"""
            <div style="
                background: #f5f7ff;
                border-left: 6px solid #667eea;
                border-radius: 12px;
                padding: 12px;
                margin-top: 8px;
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.08);
            ">
                <p style="color: #2d3748; margin: 0; font-size: 13px; line-height: 1.5;">{text_input}</p>
            </div>
            """
            st.markdown(text_card, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to analyze!")
    
    st.divider()
    st.caption("Made with Streamlit | Emotion Detection & Sentiment Analysis v1.0")

if __name__ == "__main__":
    main()