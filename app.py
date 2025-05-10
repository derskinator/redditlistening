import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import requests
import re
from datetime import datetime

st.set_page_config(page_title="Reddit Historical Listener (Pushshift)", layout="wide")
st.title("🕰️ Reddit Historical Social Listening (Pushshift API)")

# Sidebar inputs
with st.sidebar:
    st.header("Search Settings")
    query = st.text_input("Exact keyword or phrase", "donald trump")
    subreddit = st.text_input("Subreddit (e.g., 'politics')", "politics")
    result_limit = st.slider("Number of posts to fetch", 10, 500, 100)
    start_date = st.date_input("Start date", value=datetime(2023, 1, 1))
    end_date = st.date_input("End date", value=datetime.today())

# Search button
if st.button("Search Reddit (Pushshift)"):
    if not subreddit.strip():
        st.error("Please enter a subreddit name.")
        st.stop()

    # Convert to UNIX timestamps
    start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    end_ts = int(datetime.combine(end_date, datetime.max.time()).timestamp())

    st.info("Fetching data from Pushshift...")

    # Build Pushshift API request
    url = "https://api.pushshift.io/reddit/search/submission/"
    params = {
        "q": query,
        "subreddit": subreddit,
        "after": start_ts,
        "before": end_ts,
        "size": result_limit,
        "sort": "desc",
        "sort_type": "created_utc"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("data", [])

        if not results:
            st.warning("No results found for that keyword and date range.")
        else:
            pattern = rf"\b{re.escape(query.lower())}\b"
            rows = []

            for post in results:
                title = post.get("title", "")
                body = post.get("selftext", "")
                full_text = f"{title} {body}".lower()

                if re.search(pattern, full_text):
                    match_text = title if re.search(pattern, title.lower()) else body
                    sentiment = TextBlob(match_text).sentiment.polarity
                    rows.append({
                        "Date": datetime.fromtimestamp(post["created_utc"]),
                        "Subreddit": post.get("subreddit"),
                        "Title": title,
                        "Body": body[:300] + "..." if len(body) > 300 else body,
                        "Sentiment": sentiment,
                        "URL": f"https://reddit.com{post.get('permalink', '')}"
                    })

            if not rows:
                st.warning("No exact phrase matches found in title/body.")
            else:
                df = pd.DataFrame(rows)
                st.success(f"Found {len(df)} posts matching the exact phrase.")
                st.dataframe(df[["Date", "Subreddit", "Title", "Body", "Sentiment", "URL"]])

                avg_sent = df["Sentiment"].mean()
                st.metric("Average Sentiment", f"{avg_sent:.2f}")

                st.write("### Word Cloud of Matches")
                text_blob = " ".join(df["Title"].fillna("") + " " + df["Body"].fillna(""))
                wc = WordCloud(width=800, height=400, background_color="white").generate(text_blob)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, "reddit_pushshift_results.csv", "text/csv")

    except Exception as e:
        st.error(f"Pushshift API error: {e}")
