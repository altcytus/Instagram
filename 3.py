import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import openai # New: Requires 'pip install openai'

# ... [Keep your existing model training code here] ...

# 4. NEW: AI Caption Generator Function
def generate_ai_caption(topic):
    # This is a mock function. In a real app, use: 
    # client.chat.completions.create(...)
    return f"✨ Discover the magic of {topic}! 🚀 Boost your reach today. #Growth #AI"

# 5. UI Layout Updates
st.sidebar.title("🤖 AI Assistant")
topic_idea = st.sidebar.text_input("Need help? Enter a topic:")
if st.sidebar.button("Generate AI Caption"):
    ai_suggestion = generate_ai_caption(topic_idea)
    st.session_state['caption_input'] = ai_suggestion # Auto-fills the main text area

# 6. Main UI (Modified)
caption = st.text_area("Write your caption here:", 
                      value=st.session_state.get('caption_input', ""))

# 1. Load & Train (Use the expanded data I gave you earlier)
df = pd.read_csv("instagram_data.csv")
X = df[["caption_length", "hashtag_count", "sentiment_score", "post_hour"]]
y = df["engagement_rate"]
model = LinearRegression().fit(X, y)

# 2. UI Header
#st.title("📸 InstaGrowth AI Predictor")
#st.write("Predict your engagement rate before you post!")

# 3. User Inputs
caption = st.text_area("Write your caption here:")
hashtags = st.number_input("How many hashtags?", 0, 30, 5)
hour = st.slider("Posting Hour (24h format)", 0, 23, 18)

if st.button("Predict Engagement"):
    # Calculate Features
    cap_len = len(caption)
    sentiment = TextBlob(caption).sentiment.polarity  # Real AI sentiment analysis!

    # Predict
    pred = model.predict([[cap_len, hashtags, sentiment, hour]])[0]

    # Display Results
    st.metric("Predicted Engagement Rate", f"{round(pred, 2)}%")
    st.write(f"**Detected Sentiment:** {round(sentiment, 2)} (Range: -1 to 1)")

    if sentiment < 0:

        st.warning("Tip: Your caption seems a bit negative. Try adding more positive language!")
