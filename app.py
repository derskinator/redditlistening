import streamlit as st
import praw
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from datetime import datetime
import re

st.set_page_config(page_title="Reddit Social Listening", layout="wide")
st.title("🔍 Reddit Social Listening Tool")

# Sidebar inputs
with st.sidebar:
    st.header("Search Settings")
    query = st.text_input("Keyword or phrase", "walmart prices")
    subreddit = st.text_input("Subreddit (e.g., 'retail')", "retail")
    post_limit = st.slider("Number of posts to check", 10, 100, 50)
    start_date = st.date_input("Start date", value=datetime(2024, 1, 1))
    end_date = st.date_input("End date", value=datetime.today())

# Reddit API credentials
client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
user_agent = "reddit-social-listener"

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

if st.button("Search Reddit"):
    st.info("Searching posts and comments...")

    start_ts = datetime.combine(start_date, datetime.min.time()).timestamp()
    end_ts = datetime.combine(end_date, datetime.max.time()).timestamp()
    phrase = query.lower()
    pattern = rf"\b{re.escape(phrase)}\b"

    data = []
    post_count = 0

    try:
        subreddit_obj = reddit.subreddit(subreddit)
        for post in subreddit_obj.search(query, sort="new", limit=post_limit):
            post_count += 1
            created = post.created_utc
            if not (start_ts <= created <= end_ts):
                continue

            title = post.title or ""
            body = post.selftext or ""

            # Check title
            if re.search(pattern, title.lower()):
                data.append({
                    "Mention Type": "Title",
                    "Text": title,
                    "Sentiment": TextBlob(title).sentiment.polarity,
                    "Subreddit": post.subreddit.display_name,
                    "Date": datetime.fromtimestamp(created),
                    "URL": f"https://reddit.com{post.permalink}"
                })

            # Check body
            if re.search(pattern, body.lower()):
                data.append({
                    "Mention Type": "Body",
                    "Text": body[:300] + "..." if len(body) > 300 else body,
                    "Sentiment": TextBlob(body).sentiment.polarity,
                    "Subreddit": post.subreddit.display_name,
                    "Date": datetime.fromtimestamp(created),
                    "URL": f"https://reddit.com{post.permalink}"
                })

            # Check comments
            post.comments.replace_more(limit=0)
            for comment in post.comments.list():
                if re.search(pattern, (comment.body or "").lower()):
                    data.append({
                        "Mention Type": "Comment",
                        "Text": comment.body[:300] + "..." if len(comment.body) > 300 else comment.body,
                        "Sentiment": TextBlob(comment.body).sentiment.polarity,
                        "Subreddit": post.subreddit.display_name,
                        "Date": datetime.fromtimestamp(comment.created_utc),
                        "URL": f"https://reddit.com{comment.permalink}"
                    })

        if not data:
            st.warning("No matches found.")
        else:
            df = pd.DataFrame(data)
            st.success(f"Found {len(df)} matches in {post_count} posts.")
            st.dataframe(df[["Date", "Subreddit", "Mention Type", "Text", "Sentiment", "URL"]])

            # Show average sentiment
            avg_sent = df["Sentiment"].mean()
            st.metric("Average Sentiment", f"{avg_sent:.2f}")

            # Word Cloud
            st.write("### Word Cloud of Matched Text")
            text_blob = " ".join(df["Text"])
            wc = WordCloud(width=800, height=400, background_color="white").generate(text_blob)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # Export to CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "reddit_search_results.csv", "text/csv")

    except Exception as e:
        st.error(f"Reddit API error: {e}")
