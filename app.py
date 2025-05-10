import streamlit as st
import praw
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from collections import Counter
import nltk
import re

# Download stopwords
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# Initialize VADER
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Reddit Sentiment Scanner", layout="wide")
st.title("🕵️ Reddit Social Listening with VADER Sentiment")

# Sidebar inputs
with st.sidebar:
    st.header("Search Settings")
    query = st.text_input("Exact keyword or phrase", "donald trump")

    # Subreddit dropdown + custom input
    top_subreddits = [
        "askreddit", "politics", "funny", "worldnews", "gaming", "todayilearned", "science",
        "news", "aww", "technology", "movies", "explainlikeimfive", "IAmA", "pics", "music",
        "lifeprotips", "askscience", "books", "sports", "dataisbeautiful", "space", "food",
        "history", "gadgets", "television", "nosleep", "getmotivated", "personalfinance",
        "philosophy", "relationship_advice", "writingprompts", "nottheonion", "offmychest",
        "UpliftingNews", "Art", "fitness", "DIY", "Documentaries", "travel", "creepy",
        "Futurology", "OutOfTheLoop", "Tinder", "TrueOffMyChest", "Showerthoughts", "Memes",
        "Cringe", "Marketing", "Entrepreneur", "CryptoCurrency"
    ]

    subreddit_choice = st.selectbox(
        "Choose a subreddit",
        ["🔍 Custom subreddit"] + sorted(top_subreddits),
        index=sorted(top_subreddits).index("politics") + 1
    )

    if subreddit_choice == "🔍 Custom subreddit":
        subreddit = st.text_input("Enter a subreddit name (no r/)", "")
        if not subreddit.strip():
            st.warning("Please enter a subreddit or choose one from the list.")
            st.stop()
    else:
        subreddit = subreddit_choice

    post_limit = st.slider("Number of posts to scan", 10, 250, 100)
    today = datetime.today()
    min_date = today - timedelta(days=7)
    start_date = st.date_input("Start date", value=min_date, min_value=min_date, max_value=today)
    end_date = st.date_input("End date", value=today, min_value=min_date, max_value=today)

# Reddit credentials
client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
user_agent = "reddit-sentiment-listener"

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

if st.button("Search Reddit"):
    st.info("Scanning posts and comments...")

    start_ts = datetime.combine(start_date, datetime.min.time()).timestamp()
    end_ts = datetime.combine(end_date, datetime.max.time()).timestamp()
    phrase = query.lower().strip()
    pattern = rf"\b{re.escape(phrase)}\b"
    search_words = set(phrase.lower().split())

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
                excerpt = body[:300] + "..." if len(body) > 300 else body
                score = sia.polarity_scores(body)["compound"]
                match_count += 1
                data.append({
                    "Mention Type": "Body",
                    "Text": excerpt,
                    "Sentiment": score,
                    "Subreddit": post.subreddit.display_name,
                    "Date": datetime.fromtimestamp(post_time),
                    "URL": f"https://reddit.com{post.permalink}"
                })

            # Comments
            post.comments.replace_more(limit=0)
            for comment in post.comments.list():
                if not (start_ts <= comment.created_utc <= end_ts):
                    continue
                if re.search(pattern, comment.body.lower()):
                    excerpt = comment.body[:300] + "..." if len(comment.body) > 300 else comment.body
                    score = sia.polarity_scores(comment.body)["compound"]
                    match_count += 1
                    data.append({
                        "Mention Type": "Comment",
                        "Text": excerpt,
                        "Sentiment": score,
                        "Subreddit": post.subreddit.display_name,
                        "Date": datetime.fromtimestamp(comment.created_utc),
                        "URL": f"https://reddit.com{comment.permalink}"
                    })

        st.write(f"✅ Scanned {post_count} posts. Found {match_count} exact phrase matches.")

        if not data:
            st.warning("No matches found in the selected date range.")
        else:
            df = pd.DataFrame(data)
            st.success(f"Displaying {len(df)} results.")
            st.dataframe(df[["Date", "Subreddit", "Mention Type", "Text", "Sentiment", "URL"]])

            # Sentiment average
            avg_sent = df["Sentiment"].mean()
            if avg_sent >= 0.05:
                sentiment_label = "Positive"
            elif avg_sent <= -0.05:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"

            st.metric("Average Sentiment", f"{avg_sent:.2f}", delta=sentiment_label)

            # Explanation
            with st.expander("ℹ️ What does this sentiment score mean?"):
                st.markdown("""
**VADER Sentiment Analysis (Compound Score)**  
- **+0.05 to +1.0** → Positive  
- **-0.05 to +0.05** → Neutral  
- **-1.0 to -0.05** → Negative  

VADER is tuned for social media text and understands slang, emojis, caps, and punctuation.  
[Learn more →](https://arxiv.org/abs/1406.2416)
""")

            # Top 10 related words pie chart
            st.write("### Top 10 Related Words (Pie Chart)")
            all_text = " ".join(df["Text"]).lower()
            words = re.findall(r'\b[a-z]{3,}\b', all_text)
            filtered_words = [w for w in words if w not in stop_words and w not in search_words]
            top_words = Counter(filtered_words).most_common(10)

            if top_words:
                labels, counts = zip(*top_words)
                fig, ax = plt.subplots()
                ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=140)
                ax.axis("equal")
                st.pyplot(fig)
            else:
                st.info("Not enough related words to generate pie chart.")

            # CSV download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "reddit_sentiment_results.csv", "text/csv")

    except Exception as e:
        st.error(f"Reddit API error: {e}")
