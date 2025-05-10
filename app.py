import streamlit as st
import praw
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import nltk
import re

# Download VADER lexicon silently
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# Streamlit config
st.set_page_config(page_title="Reddit Listener (VADER)", layout="wide")
st.title("🕵️ Reddit Social Listening with Improved Sentiment (VADER)")

# Sidebar inputs
with st.sidebar:
    st.header("Search Settings")
    query = st.text_input("Exact keyword or phrase", "donald trump")
    subreddit = st.text_input("Subreddit (e.g., 'politics')", "politics")
    post_limit = st.slider("Number of recent posts to scan", 10, 250, 100)

    today = datetime.today()
    min_date = today - timedelta(days=7)
    start_date = st.date_input("Start date (max 7 days ago)", value=min_date, min_value=min_date, max_value=today)
    end_date = st.date_input("End date", value=today, min_value=min_date, max_value=today)

# Reddit credentials
client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
user_agent = "reddit-social-listener"

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

if st.button("Search Reddit"):
    if not subreddit.strip():
        st.error("Please enter a subreddit (e.g., 'news', 'politics').")
        st.stop()

    st.info("Scanning posts and comments using VADER sentiment...")

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

            # Title match
            if re.search(pattern, title.lower()):
                score = sia.polarity_scores(title)["compound"]
                match_count += 1
                data.append({
                    "Mention Type": "Title",
                    "Text": title,
                    "Sentiment": score,
                    "Subreddit": post.subreddit.display_name,
                    "Date": datetime.fromtimestamp(post_time),
                    "URL": f"https://reddit.com{post.permalink}"
                })

            # Body match
            if re.search(pattern, body.lower()):
                text_excerpt = body[:300] + "..." if len(body) > 300 else body
                score = sia.polarity_scores(body)["compound"]
                match_count += 1
                data.append({
                    "Mention Type": "Body",
                    "Text": text_excerpt,
                    "Sentiment": score,
                    "Subreddit": post.subreddit.display_name,
                    "Date": datetime.fromtimestamp(post_time),
                    "URL": f"https://reddit.com{post.permalink}"
                })

            # Comment match
            post.comments.replace_more(limit=0)
            for comment in post.comments.list():
                if not (start_ts <= comment.created_utc <= end_ts):
                    continue
                if re.search(pattern, comment.body.lower()):
                    text_excerpt = comment.body[:300] + "..." if len(comment.body) > 300 else comment.body
                    score = sia.polarity_scores(comment.body)["compound"]
                    match_count += 1
                    data.append({
                        "Mention Type": "Comment",
                        "Text": text_excerpt,
                        "Sentiment": score,
                        "Subreddit": post.subreddit.display_name,
                        "Date": datetime.fromtimestamp(comment.created_utc),
                        "URL": f"https://reddit.com{comment.permalink}"
                    })

        st.write(f"✅ Scanned {post_count} posts. Found {match_count} matches.")

        if not data:
            st.warning("No exact matches found in this 7-day window.")
        else:
            df = pd.DataFrame(data)
            st.success(f"Displaying {len(df)} matches.")
            st.dataframe(df[["Date", "Subreddit", "Mention Type", "Text", "Sentiment", "URL"]])

            avg_sent = df["Sentiment"].mean()
            st.metric("Average Sentiment (VADER)", f"{avg_sent:.2f}")

            # Word Cloud
            st.write("### Word Cloud of Matched Text")
            all_text = " ".join(df["Text"])
            wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # Export
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "reddit_vader_sentiment.csv", "text/csv")

    except Exception as e:
        st.error(f"Reddit API error: {e}")


