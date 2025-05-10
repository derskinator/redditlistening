import streamlit as st
import praw
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from datetime import datetime

st.set_page_config(page_title="Reddit Social Listening", layout="wide")
st.title("🔍 Reddit Social Listening Tool")

# Sidebar inputs
with st.sidebar:
    st.header("Search Settings")
    query = st.text_input("Keyword or phrase", "walmart")
    subreddit = st.text_input("Subreddit (optional)", "")
    post_limit = st.slider("Number of posts to fetch", 10, 100, 50)
    start_date = st.date_input("Start date", value=datetime(2024, 1, 1))
    end_date = st.date_input("End date", value=datetime.today())

# Reddit credentials
client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
user_agent = "reddit-social-listener"

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# Trigger search
if st.button("Search Reddit"):
    st.info("Searching Reddit...")

    try:
        subreddit_obj = reddit.subreddit(subreddit) if subreddit else reddit.subreddit("all")
        posts = subreddit_obj.search(query, sort="new", limit=post_limit)

        start_ts = datetime.combine(start_date, datetime.min.time()).timestamp()
        end_ts = datetime.combine(end_date, datetime.max.time()).timestamp()

        data = []
        checked = 0
        matched = 0

        for post in posts:
            checked += 1
            title = post.title or ""
            if start_ts <= post.created_utc <= end_ts:
                matched += 1
                sentiment = TextBlob(title).sentiment.polarity
                data.append({
                    "Title": title,
                    "Sentiment": sentiment,
                    "Subreddit": post.subreddit.display_name,
                    "Date": datetime.fromtimestamp(post.created_utc),
                    "URL": f"https://reddit.com{post.permalink}"
                })

        st.write(f"✅ Checked {checked} posts. Found {matched} in date range.")

        if not data:
            st.warning("No posts found for that keyword in the selected date range.")
        else:
            df = pd.DataFrame(data)
            st.success(f"Displaying {len(df)} posts.")
            st.dataframe(df)

            # Average sentiment
            avg_sent = df["Sentiment"].mean()
            st.metric("Average Sentiment", f"{avg_sent:.2f}")

            # Word Cloud
            st.write("### Word Cloud of Post Titles")
            wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df["Title"]))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # CSV export
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "reddit_search_results.csv", "text/csv")

    except Exception as e:
        st.error(f"Reddit API error: {e}")
