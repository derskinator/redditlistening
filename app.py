import requests
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import streamlit as st
from datetime import datetime

st.title("🔍 Reddit Social Listening Tool")

query = st.text_input("Enter keyword or phrase", "rinsekit")
subreddit = st.text_input("Subreddit (optional)", "camping")
start_date = st.date_input("Start date", value=pd.to_datetime("2024-01-01"))
end_date = st.date_input("End date", value=datetime.today())

start_utc = int(pd.Timestamp(start_date).timestamp())
end_utc = int(pd.Timestamp(end_date).timestamp())

def fetch_reddit_posts(query, subreddit, start_utc, end_utc, size=100):
    base_url = "https://api.pushshift.io/reddit/search/submission/"
    params = {
        "q": query,
        "subreddit": subreddit if subreddit else None,
        "after": start_utc,
        "before": end_utc,
        "sort": "desc",
        "size": size
    }
    response = requests.get(base_url, params=params)
    return response.json().get("data", []) if response.status_code == 200 else []

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def generate_wordcloud(texts):
    combined_text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

if st.button("Search"):
    posts = fetch_reddit_posts(query, subreddit, start_utc, end_utc)

    if posts:
        data = []
        for post in posts:
            title = post.get("title", "")
            sentiment = get_sentiment(title)
            data.append({
                "Title": title,
                "Subreddit": post.get("subreddit"),
                "Date": pd.to_datetime(post["created_utc"], unit="s"),
                "Sentiment": sentiment,
                "URL": post.get("full_link", "")
            })

        df = pd.DataFrame(data)
        st.write("### Search Results", df)

        avg_sentiment = df["Sentiment"].mean()
        st.metric("Average Sentiment", f"{avg_sentiment:.2f}")

        st.write("### Word Cloud")
        generate_wordcloud(df["Title"].tolist())
    else:
        st.warning("No posts found for the given query and date range.")
