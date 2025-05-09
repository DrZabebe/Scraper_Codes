import praw
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta, timezone
import snowflake.connector

# Your Reddit API credentials
REDDIT_CLIENT_ID = 'your_client_id'
REDDIT_CLIENT_SECRET = 'your_client_secret'
REDDIT_USER_AGENT = 'your_user_agent'
SNOWFLAKE_ACCOUNT = 'your_account'
SNOWFLAKE_USER = 'your_user'
SNOWFLAKE_PASSWORD = 'your_password'
SNOWFLAKE_WAREHOUSE = 'your_warehouse'
SNOWFLAKE_DATABASE = 'your_database'
SNOWFLAKE_SCHEMA = 'your_schema'
SNOWFLAKE_TABLE = 'your_table'

# Keywords to search for
KEYWORDS = [
    "AI in healthcare", "Doctor AI", "primary care AI tools",
    "AI bias healthcare", "AI trust doctors", "AI replacing doctors",
    "machine learning in healthcare", "healthcare automation AI",
    "electronic medical records AI"
]
DAYS_BACK = 730  # 2 years
MAX_POSTS_PER_KEYWORD = 500

# -------------------- FUNCTIONS --------------------

def reddit_connect():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase and remove extra spaces
    return text

def is_relevant_post(title, body):
    # Filter out irrelevant posts based on specific keywords
    text = (title + " " + body).lower()
    if any(x in text for x in ["rash", "staph", "folliculitis", "bleach", "detergent", "pajamas", "tape", "shower", "spray"]):
        return False
    if not any(x in text for x in ["ai", "machine learning", "artificial intelligence"]):
        return False
    if not any(x in text for x in ["healthcare", "doctor", "physician", "medical", "clinic", "primary care"]):
        return False
    word_count = len(text.split())
    return 30 <= word_count <= 300  # Ensure the post is within 30-300 words

def search_reddit_posts(reddit, keywords, days_back, max_posts_per_keyword):
    all_posts = []
    cutoff_timestamp = datetime.now(timezone.utc) - timedelta(days=days_back)
    for keyword in keywords:
        print(f"\n🔍 Searching for keyword: '{keyword}'...")
        count = 0
        for submission in reddit.subreddit("all").search(keyword, sort='new', limit=max_posts_per_keyword):
            created = datetime.fromtimestamp(submission.created_utc, timezone.utc)
            if created < cutoff_timestamp:
                continue
            if not is_relevant_post(submission.title, submission.selftext):
                continue
            all_posts.append({
                "Author": str(submission.author),
                "Title": submission.title,
                "Body": submission.selftext,
                "CreatedUTC": created.strftime('%Y-%m-%d %H:%M:%S'),
                "Subreddit": submission.subreddit.display_name,
                "URL": submission.url,
                "Score": submission.score,
                "Comments": submission.num_comments,
                "Keyword": keyword
            })
            count += 1
        print(f"✅ Collected {count} filtered posts for keyword '{keyword}'")
    return pd.DataFrame(all_posts)

def analyze_sentiment(df):
    sid = SentimentIntensityAnalyzer()
    sentiments = []
    for _, row in df.iterrows():
        text = clean_text(f"{row['Title']} {row['Body']}")
        score = sid.polarity_scores(text)
        if score["compound"] >= 0.05:
            sentiments.append("Positive")
        elif score["compound"] <= -0.05:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    df["Sentiment"] = sentiments
    return df

def snowflake_connect():
    return snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )

def insert_to_snowflake(df, table_name):
    try:
        conn = snowflake_connect()
        cs = conn.cursor()
        print(f"✅ Connected to Snowflake. Uploading {len(df)} records...")
        insert_sql = f"""
        INSERT INTO {table_name} 
        ("AUTHOR", "TITLE", "BODY", "CREATEDUTC", "SUBREDDIT", "URL", "SCORE", "COMMENTS", "SENTIMENT", "KEYWORD")
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        for _, row in df.iterrows():
            cs.execute(insert_sql, (
                row["Author"], row["Title"], row["Body"], row["CreatedUTC"],
                row["Subreddit"], row["URL"], row["Score"], row["Comments"],
                row["Sentiment"], row["Keyword"]
            ))
        cs.close()
        conn.close()
        print(f"✅ Upload complete to Snowflake table: {table_name}")
    except Exception as e:
        print(f"❌ Snowflake Upload Error: {e}")

# -------------------- MAIN --------------------
if __name__ == "__main__":
    reddit = reddit_connect()
    posts_df = search_reddit_posts(reddit, KEYWORDS, DAYS_BACK, MAX_POSTS_PER_KEYWORD)
    if not posts_df.empty:
        posts_df = analyze_sentiment(posts_df)
        insert_to_snowflake(posts_df, SNOWFLAKE_TABLE)
        posts_df.to_csv("reddit_sentiment_cleaned.csv", index=False)
        print("✅ Saved to reddit_sentiment_cleaned.csv")
    else:
        print("⚠️ No relevant posts found.")
