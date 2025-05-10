import streamlit as st
import praw
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from datetime import datetime
import re

# Streamlit config
st.set_page_config(page_title="Reddit Social Listening", layout="wide")
st.title("🔍 Reddit Social Listening Tool")

# Sidebar filters
with st.sidebar:
    st.header("Search Settings")
    query = st.text_input("Keyword or phrase (exact match)", "donald trump")
    subreddit = st.text_input("Subreddit (e.g., 'politics')", "politics")
    post_limit = st.slider("Number of recent posts to scan", 10, 250, 100)
    start_date = st.date_input("Start date", value=datetime(2024, 1, 1))
    end_date = st.date_input("End date", value=datetime.today())

# Reddit API credentials
client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
user_agent = "reddit-social-listener"

# Initialize Reddit
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

if st.button("Search Reddit"):
    # Validate subreddit input
    if not subreddit.strip():
        st.error("Please enter a subreddit (e.g., 'news', 'politics', 'retail'). Reddit does not support searching across all subreddits.")
        st.stop()

    st.info("Fetching posts and checking for exact phrase matches...")

    start_ts = datetime.combine(start_date, datetime.min.time()).timestamp()
    end_ts = datetime.combine(end_date, datetime.max.time()).timestamp()
    phrase = query.lower().strip()
    pattern = rf"\b{re.escape(phrase)}\b"

    data = []
    post_count = 0
    match_count = 0

    try:
        subreddit_obj = reddit.subreddit(subreddit)
        for post in subreddit_obj.new(limit=post_limit):
            post_count += 1
            title = post.title or ""
            body = post.selftext or ""
            post_time = post.created_utc

            if not (start_ts <= post_time <= end_ts):
                continue

            # Match: title
            if re.search(pattern, title.lower()):
                match_count += 1
                data.append({
                    "Mention Type": "Title",
                    "Text": title,
                    "Sentiment": TextBlob(title).sentiment.polarity,
                    "Subreddit": post.subreddit.display_name,
                    "Date": datetime.fromtimestamp(post_time),
                    "URL": f"https://reddit.com{post.permalink}"
                })

            # Match: body
            if re.search(pattern, body.lower()):
                match_count += 1
                data.append({
                    "Mention Type": "Body",
                    "Text": body[:300] + "..." if len(body) > 300 else body,
                    "Sentiment": TextBlob(body).sentiment.polarity,
                    "Subreddit": post.subreddit.display_name,
                    "Date": datetime.fromtimestamp(post_time),
                    "URL": f"https://reddit.com{post.permalink}"
                })

            # Match: comments
            post.comments.replace_more(limit=0)
            for comment in post.comments.list():
                if not (start_ts <= comment.created_utc <= end_ts):
                    continue
                if re.search(pattern, comment.body.lower()):
                    match_count += 1
                    data.append({
                        "Mention Type": "Comment",
                        "Text": comment.body[:300] + "..." if len(comment.body) > 300 else comment.body,
                        "Sentiment": TextBlob(comment.body).sentiment.polarity,
                        "Subreddit": post.subreddit.display_name,
                        "Date": datetime.fromtimestamp(comment.created_utc),
                        "URL": f"https://reddit.com{comment.permalink}"
                    })

        st.write(f"✅ Scanned {post_count} posts. Found {match_count} exact matches.")

        if not data:
            st.warning("No matches found for the exact phrase in this date range.")
        else:
            df = pd.DataFrame(data)
            st.success(f"Displaying {len(df)} total matches.")
            st.dataframe(df[["Date", "Subreddit", "Mention Type", "Text", "Sentiment", "URL"]])

            # Sentiment average
            avg_sent = df["Sentiment"].mean()
            st.metric("Average Sentiment", f"{avg_sent:.2f}")

            # Word cloud
            st.write("### Word Cloud of Matched Text")
            wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df["Text"]))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # CSV download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "reddit_search_results.csv", "text/csv")

    except Exception as e:
        st.error(f"Reddit API error: {e}")
