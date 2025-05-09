# -------------------- IMPORTS --------------------
from Bio import Entrez, Medline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import pandas as pd
import snowflake.connector
import os

# -------------------- CONFIG --------------------

Entrez.email = "gzne.2021@gmail.com"
MAX_PER_BATCH = 1000  # Entrez safe default
TOTAL_TO_FETCH = 5000  # Increase this if you want more


# Snowflake credentials
SNOWFLAKE_USER = ''
SNOWFLAKE_PASSWORD = ''
SNOWFLAKE_ACCOUNT = ''
SNOWFLAKE_WAREHOUSE = ''
SNOWFLAKE_DATABASE = ''
SNOWFLAKE_SCHEMA = ''
SNOWFLAKE_TABLE = ''

TODAY = datetime.today().strftime("%Y-%m-%d")

barrier_examples = {

"Data Quality and Integrity Issues": "Incomplete, inconsistent, or biased healthcare data undermines AI system reliability, leading to incorrect recommendations and clinician mistrust."

}


# -------------------- FUNCTIONS --------------------
def snowflake_connect():
    ctx = snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )
    return ctx


def insert_to_snowflake(df, table_name):
    try:
        conn = snowflake_connect()
        cs = conn.cursor()
        print(f"✅ Connected to Snowflake. Uploading {len(df)} records...")

        # Create the insert SQL
        for index, row in df.iterrows():
            sql = f"""
            INSERT INTO {table_name} ("AUTHOR", "TITLE", "GAPS", "CATEGORY", "YEAR", "LINK")
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cs.execute(sql, (
                row["Author(s)"],
                row["Title"],
                row["Gaps"],
                row["Category"],
                row["Year"],
                row["Link"]
            ))

        cs.close()
        conn.close()
        print(f"✅ Upload complete into {table_name}.\n")

    except Exception as e:
        print(f"❌ Snowflake Upload Error: {e}")


def search_pubmed(query, retstart=0, retmax=1000):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax, retstart=retstart)
    record = Entrez.read(handle)
    return record["IdList"]


def fetch_details(id_list):
    ids = ",".join(id_list)
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
    records = Medline.parse(handle)
    articles = []

    for record in records:
        title = record.get("TI", "No Title")
        authors = "; ".join(record.get("AU", ["No Author"]))
        abstract = record.get("AB", "")
        combined_text = f"{title} {abstract}"
        pub_year = record.get("DP", "Unknown")[:4]
        pmid = record.get("PMID", "0")
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        articles.append({
            "Author(s)": authors,
            "Title": title,
            "Text": combined_text,
            "Year": pub_year,
            "Link": url
        })
    return articles


def classify_articles_nlp(articles, barrier_examples):
    barrier_names = list(barrier_examples.keys())
    barrier_texts = list(barrier_examples.values())

    all_texts = [article["Text"] for article in articles] + barrier_texts
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    article_vectors = tfidf_matrix[:-len(barrier_texts)]
    category_vectors = tfidf_matrix[-len(barrier_texts):]

    similarities = cosine_similarity(article_vectors, category_vectors)
    categories = [barrier_names[i.argmax()] for i in similarities]

    for idx, article in enumerate(articles):
        article["Category"] = categories[idx]
        article["Gaps"] = article["Text"][:400]
        del article["Text"]
    return articles


# -------------------- MAIN PAGINATION WORKFLOW --------------------
def run_batched(query, total_to_fetch=1500, batch_size=500):
    for start in range(0, total_to_fetch, batch_size):
        print(f"\n🔄 Fetching records {start + 1} to {start + batch_size}")
        ids = search_pubmed(query, retstart=start, retmax=batch_size)
        if not ids:
            print("⚠️ No more records returned.")
            break

        raw_articles = fetch_details(ids)
        classified = classify_articles_nlp(raw_articles, barrier_examples)

        df = pd.DataFrame(classified)[["Author(s)", "Title", "Gaps", "Category", "Year", "Link"]]

        # Direct insert into Snowflake
        insert_to_snowflake(df, SNOWFLAKE_TABLE)


# -------------------- RUN IT --------------------
query = '("artificial intelligence" OR "AI") AND ("primary care" OR "healthcare") AND ("adoption" OR "implementation")'
run_batched(query, total_to_fetch=20000, batch_size=1000)
