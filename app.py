import streamlit as st
import praw
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from datetime import datetime

# Streamlit config
st.set_page_config(page_title="Reddit Social Listening", layout="wide")
st.title("🔍 Reddit Social Listening Tool")

# Sidebar controls
with st.sidebar:
    st.header("Search Settings")
    query = st.text_input("Keyword or phrase", "rinsekit")
    subreddit = st.text_input("Subreddit (optional)", "camping")
    post_limit = st.slider("Max posts to check", 10, 500, 100)
    start_date = st.date_input("Start date", value=datetime(2024, 1, 1))
    end_date = st.date_input("End date", value=datetime.today())

# Reddit credentials
client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
user_agent = "reddit-social-listener"

# Reddit client
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# Main logic
if st.button("Search Reddit"):
    st.info("Searching Reddit...")

    try:
        subreddit_obj = reddit.subreddit(subreddit) if subreddit else reddit.subreddit("all")
        posts = subreddit_obj.new(limit=post_limit)  # Get latest N posts

        start_timestamp = datetime.combine(start_date, datetime.min.time()).timestamp()
        end_timestamp = datetime.combine(end_date, datetime.max.time()).timestamp()

        data = []
        for post in posts:
            title = post.title or ""
            created = post.created_utc
            if (
                query.lower() in title.lower() and
                start_timestamp <= created <= end_timestamp
            ):
                sentiment = TextBlob(title).sentiment.polarity
                data.append({
                    "Title": title,
                    "Sentiment": sentiment,
                    "Subreddit": post.subreddit.display_name,
                    "Date": datetime.fromtimestamp(created),
                    "URL": f"https://reddit.com{post.permalink}"
                })

        if not data:
            st.warning("No matching posts found in selected date range.")
        else:
            df = pd.DataFrame(data)
            st.success(f"Found {len(df)} matching posts.")
            st.dataframe(df)

            # Sentiment
            avg_sent = df["Sentiment"].mean()
            st.metric("Average Sentiment", f"{avg_sent:.2f}")

            # Word Cloud
            st.write("### Word Cloud of Post Titles")
            wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df["Title"]))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # CSV Export
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "reddit_search_results.csv", "text/csv")

    except Exception as e:
        st.error(f"Reddit API error: {e}")
