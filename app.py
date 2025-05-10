import streamlit as st
import praw
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from datetime import datetime

# Streamlit page config
st.set_page_config(page_title="Reddit Social Listening", layout="wide")
st.title("🔍 Reddit Social Listening Tool")

# Input form
with st.sidebar:
    st.header("Search Settings")
    query = st.text_input("Keyword or phrase", "rinsekit")
    subreddit = st.text_input("Subreddit (optional)", "camping")
    post_limit = st.slider("Number of posts to fetch", 10, 100, 50)

# Load Reddit credentials from secrets
client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
user_agent = "reddit-social-listener"

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# Search and analyze
if st.button("Search Reddit"):
    st.info("Fetching posts...")

    try:
        subreddit_obj = reddit.subreddit(subreddit) if subreddit else reddit.subreddit("all")
        posts = subreddit_obj.search(query, sort="new", limit=post_limit)
        data = []

        for post in posts:
            title = post.title
            sentiment = TextBlob(title).sentiment.polarity
            data.append({
                "Title": title,
                "Sentiment": sentiment,
                "Subreddit": post.subreddit.display_name,
                "Date": datetime.fromtimestamp(post.created_utc),
                "URL": f"https://reddit.com{post.permalink}"
            })

        if not data:
            st.warning("No posts found.")
        else:
            df = pd.DataFrame(data)
            st.success(f"Found {len(df)} posts.")
            st.dataframe(df)

            # Sentiment average
            avg_sent = df["Sentiment"].mean()
            st.metric("Average Sentiment", f"{avg_sent:.2f}")

            # Word Cloud
            st.write("### Word Cloud of Post Titles")
            text = " ".join(df["Title"])
            wc = WordCloud(width=800, height=400, background_color="white").generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # Download CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "reddit_search_results.csv", "text/csv")

    except Exception as e:
        st.error(f"Reddit API error: {e}")
