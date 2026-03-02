import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import google.generativeai as genai
import nltk

# 1. INITIAL SETUP & DATA
@st.cache_resource
def setup_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')

setup_nltk()

# Setup Gemini
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error("Gemini API Key missing in Secrets!")
    gemini_model = None

# Train the Predictive Model
df = pd.read_csv("instagram_data.csv")
X = df[["caption_length", "hashtag_count", "sentiment_score", "post_hour"]]
y = df["engagement_rate"]
model = LinearRegression().fit(X, y)

# 2. HELPER FUNCTIONS
def generate_ai_caption(topic):
    if gemini_model:
        try:
            prompt = f"Write a catchy Instagram caption about {topic}. Keep it short and include 3 hashtags."
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"
    return ""

# 3. UI LAYOUT
st.title("📸 InstaGrowth AI Predictor")
st.write("Generate AI captions and predict your engagement rate instantly.")

# Sidebar for AI Generation
st.sidebar.title("🤖 Gemini AI Assistant")
topic_idea = st.sidebar.text_input("Enter a topic (e.g., Summer Travel):")

if st.sidebar.button("Generate with Gemini"):
    if topic_idea:
        with st.sidebar.spinner("Gemini is writing..."):
            ai_suggestion = generate_ai_caption(topic_idea)
            st.session_state['caption_input'] = ai_suggestion
    else:
        st.sidebar.warning("Please enter a topic first.")

st.divider()

# Main Input Section
# Using st.session_state to link the AI result to the text box
default_text = st.session_state.get('caption_input', "")
caption = st.text_area("Finalize your caption here:", value=default_text, height=150)

col1, col2 = st.columns(2)
with col1:
    hashtags = st.number_input("How many hashtags?", 0, 30, 5)
with col2:
    hour = st.slider("Posting Hour (24h)", 0, 23, 18)

# 4. PREDICTION LOGIC
if st.button("Predict Engagement", type="primary"):
    if caption:
        # Feature Engineering
        cap_len = len(caption)
        sentiment = TextBlob(caption).sentiment.polarity
        
        # ML Prediction
        pred = model.predict([[cap_len, hashtags, sentiment, hour]])[0]
        
        # Display Results
        st.subheader("Analysis Results")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Predicted Engagement", f"{round(max(0, pred), 2)}%")
        m_col2.metric("Sentiment Score", f"{round(sentiment, 2)}")

        if sentiment < 0:
            st.info("💡 **AI Tip:** This caption sounds a bit negative. Positive captions usually get 15% more likes!")
        else:
            st.success("✨ Great tone! This post looks ready to go.")
    else:
        st.error("Please write or generate a caption first.")
