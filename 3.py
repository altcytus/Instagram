import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import openai  # <--- It's back!
import os

# 1. SETUP OPENAI (You need an API key from platform.openai.com)
# For a school project, you can hardcode it or use an environment variable
os.environ["OPENAI_API_KEY"] = "your-api-key-here" 
client = openai.OpenAI()

# 2. LOAD & TRAIN PREDICTIVE MODEL
df = pd.read_csv("instagram_data.csv")
X = df[["caption_length", "hashtag_count", "sentiment_score", "post_hour"]]
y = df["engagement_rate"]
model = LinearRegression().fit(X, y)

# 3. GENERATIVE AI FUNCTION (The "Integration")
def generate_ai_caption(topic):
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Or gpt-3.5-turbo for cheaper testing
            messages=[
                {"role": "system", "content": "You are an Instagram marketing expert."},
                {"role": "user", "content": f"Write a high-engagement caption about: {topic}. Keep it under 150 characters."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}. (Make sure your API key is set!)"

# 4. UI LAYOUT
st.title("📸 InstaGrowth AI Predictor")

# Sidebar for the AI Assistant
st.sidebar.title("🤖 AI Content Creator")
topic_idea = st.sidebar.text_input("What is your post about?")

if st.sidebar.button("Generate AI Caption"):
    with st.sidebar.spinner("AI is thinking..."):
        ai_suggestion = generate_ai_caption(topic_idea)
        st.session_state['caption_input'] = ai_suggestion 

# 5. MAIN INPUTS
# This 'value' link is what makes the AI-generated text appear in the box
default_text = st.session_state.get('caption_input', "")
caption = st.text_area("Write your caption here:", value=default_text)
hashtags = st.number_input("How many hashtags?", 0, 30, 5)
hour = st.slider("Posting Hour (24h format)", 0, 23, 18)

# 6. PREDICTION
if st.button("Predict Engagement"):
    cap_len = len(caption)
    sentiment = TextBlob(caption).sentiment.polarity
    pred = model.predict([[cap_len, hashtags, sentiment, hour]])[0]

    st.metric("Predicted Engagement Rate", f"{round(max(0, pred), 2)}%")
    st.write(f"**Sentiment:** {round(sentiment, 2)}")
        if sentiment < 0:
            st.info("💡 **AI Tip:** Posts with positive sentiment usually perform 15% better in this niche.")
