import requests
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Reddit Social Listening", layout="wide")
st.title("🔍 Reddit Social Listening Tool")

# Inputs
query = st.text_input("Enter keyword or phrase", "rinsekit")
subreddit = st.text_input("Subreddit (optional)", "camping")
start_date = st.date_input("Start date", value=datetime(2024, 1, 1))
end_date = st.date_input("End date", value=datetime.today())

# Convert to timestamps
start_utc = int(pd.Timestamp(start_date).timestamp())
end_utc = int(pd.Timestamp(end_date).timestamp())

# Search button
if st.button("Search"):

    # Request data from Pushshift
    base_url = "https://api.pushshift.io/reddit/search/submission/"
    params = {
        "q": query,
        "after": start_utc,
        "before": end_utc,
        "subreddit": subreddit if subreddit else None,
        "sort": "desc",
        "size": 100
    }

    st.info("Fetching data from Reddit...")
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json().get("data", [])
    except Exception as e:
        st.error(f"API error: {e}")
        data = []

    if not data:
        st.warning("No posts found. Try a broader keyword or remove the subreddit.")
    else:
        posts = []
        for post in data:
            title = post.get("title", "")
            sentiment = TextBlob(title).sentiment.polarity
            posts.append({
                "Title": title,
                "Sentiment": sentiment,
                "Subreddit": post.get("subreddit", ""),
                "Date": pd.to_datetime(post.get("created_utc", 0), unit="s"),
                "URL": post.get("full_link", "")
            })

        df = pd.DataFrame(posts)
        st.success(f"Found {len(df)} posts.")
        st.dataframe(df)

        avg_sentiment = df["Sentiment"].mean()
        st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")

        st.write("### Word Cloud of Post Titles")
        combined_text = " ".join(df["Title"])
        wc = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
