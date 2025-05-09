import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import snowflake.connector
import time
import json

# -------------------- CONFIGURATION --------------------
# Snowflake credentials
SNOWFLAKE_USER = ''
SNOWFLAKE_PASSWORD = ''
SNOWFLAKE_ACCOUNT = ''
SNOWFLAKE_WAREHOUSE = ''
SNOWFLAKE_DATABASE = ''
SNOWFLAKE_SCHEMA = ''
SNOWFLAKE_TABLE = 'elsevier_articles_staging'

# Elsevier API Key
ELSEVIER_API_KEY = ""  # Your 32-character key

# Barrier categories
barrier_examples = {
    "Trust and Lack of Transparency": "AI systems are opaque, black-box models reduce trust among clinicians.",
    "Loss of Autonomy and Clinical Authority": "AI systems may override clinical decisions, raising autonomy concerns.",
    "Training Deficits and Cognitive Burden": "Clinicians lack AI training, causing cognitive overload and low confidence.",
    "System Design Failures and Workflow Disruption": "Poor AI integration disrupts workflows and reduces usability.",
    "Organizational Readiness and Structural Constraints": "Many practices lack infrastructure, strategy, or leadership support.",
    "Socio-Cultural and Patient Interaction Barriers": "Cultural bias and fear of dehumanized care impact adoption.",
    "Data Quality and Integrity Issues": "Incomplete, inconsistent, or biased healthcare data undermines AI system reliability."
}


# -------------------- FUNCTIONS --------------------

# Connect to Snowflake
def snowflake_connect():
    return snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )


# Insert DataFrame into Snowflake
def insert_to_snowflake(df, table_name):
    try:
        conn = snowflake_connect()
        cs = conn.cursor()
        print(f"✅ Connected to Snowflake. Uploading {len(df)} records...")

        insert_sql = f"""
            INSERT INTO {table_name} ("AUTHOR", "TITLE", "GAPS", "CATEGORY", "YEAR", "LINK")
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        for _, row in df.iterrows():
            cs.execute(insert_sql, (
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


# Search for articles using only authorized parameters
def search_articles(query, start=0, count=25):
    """Search for articles and return the data directly"""
    url = "https://api.elsevier.com/content/search/scopus"
    headers = {
        'X-ELS-APIKey': ELSEVIER_API_KEY,
        'Accept': 'application/json'
    }
    # Using only parameters that work with your API key
    params = {
        'query': query,
        'count': count,
        'start': start,
        'date': '2010-2024',  # More relevant recent articles
        'sort': 'relevancy'  # Get most relevant results first
    }

    print(f"🔍 Searching for articles (batch starting at {start})...")
    try:
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            entries = data.get('search-results', {}).get('entry', [])
            total = data.get('search-results', {}).get('opensearch:totalResults', '0')
            print(f"📊 Found {len(entries)} articles (out of {total} total)")

            articles = []
            for entry in entries:
                # Extract article data from search response
                try:
                    # Note: Some fields may not be available due to API access limitations
                    author = entry.get('dc:creator', 'Unknown Author')
                    title = entry.get('dc:title', 'No Title')

                    # Extract year from date
                    date = entry.get('prism:coverDate', '')
                    year = date[:4] if date else entry.get('prism:coverDisplayDate', '')[:4]
                    if not year:
                        year = 'Unknown'

                    # Get abstract if available
                    abstract = entry.get('dc:description', '')

                    # Get DOI and link
                    doi = entry.get('prism:doi', '')
                    # Get link to abstract
                    link = None
                    if 'link' in entry and isinstance(entry['link'], list):
                        for link_item in entry['link']:
                            if link_item.get('@ref') == 'scopus':
                                link = link_item.get('@href', '')
                                break

                    if not link and doi:
                        link = f"https://doi.org/{doi}"
                    elif not link:
                        link = "No Link Available"

                    # Combine title and abstract for text classification
                    text = f"{title} {abstract}"

                    article = {
                        "Author(s)": author,
                        "Title": title,
                        "Text": text,
                        "Year": year,
                        "Link": link,
                        "DOI": doi
                    }
                    articles.append(article)

                except Exception as e:
                    print(f"⚠️ Error processing entry: {e}")

            return articles

        else:
            print(f"❌ Error searching articles: {response.status_code}")
            print(response.text[:200])
            return []

    except Exception as e:
        print(f"❌ Error in search_articles: {e}")
        return []


# Get detailed article information with backup sources
def enrich_article_details(articles):
    """Try to get more details about articles from Abstract API if available"""
    if not articles:
        return articles

    enriched_articles = []

    for i, article in enumerate(articles):
        doi = article.get('DOI')

        if doi and doi != 'No DOI':
            print(f"🔍 Enriching article {i + 1}/{len(articles)}: {doi}")

            # Try to get more details from abstract API
            url = f"https://api.elsevier.com/content/abstract/doi/{doi}"
            headers = {
                'X-ELS-APIKey': ELSEVIER_API_KEY,
                'Accept': 'application/json'
            }

            try:
                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    if 'abstracts-retrieval-response' in data:
                        coredata = data['abstracts-retrieval-response'].get('coredata', {})

                        # Get better abstract if available
                        if 'dc:description' in coredata:
                            # Update the text with better abstract
                            article['Text'] = article['Title'] + ' ' + coredata['dc:description']

                        # Get better author information if available
                        if 'authors' in data['abstracts-retrieval-response']:
                            authors_data = data['abstracts-retrieval-response']['authors'].get('author', [])
                            if authors_data and isinstance(authors_data, list):
                                author_names = []
                                for author in authors_data[:5]:  # Limit to first 5 authors
                                    if 'preferred-name' in author:
                                        surname = author['preferred-name'].get('surname', '')
                                        given_name = author['preferred-name'].get('given-name', '')
                                        if surname or given_name:
                                            author_names.append(f"{surname}, {given_name}")

                                if author_names:
                                    article['Author(s)'] = "; ".join(author_names)
                                    if len(authors_data) > 5:
                                        article['Author(s)'] += " et al."

                else:
                    print(f"⚠️ Could not enrich article {doi}: {response.status_code}")

            except Exception as e:
                print(f"⚠️ Error enriching article {doi}: {e}")

            # Add small delay to avoid rate limiting
            time.sleep(0.5)

        enriched_articles.append(article)

    return enriched_articles


# Classify articles into barrier categories
def classify_articles_nlp(articles, barrier_examples):
    if not articles:
        return []

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


# Main runner with batching
def run_batched_search(query, total_to_fetch=50, batch_size=25):
    all_articles = []

    for start in range(0, total_to_fetch, batch_size):
        print(f"\n🔄 Fetching batch from {start + 1} to {min(start + batch_size, total_to_fetch)}")

        # Add rate limiting to avoid API rate limits
        if start > 0:
            print("⏱️ Waiting 1 second before next batch...")
            time.sleep(1)

        # Get articles directly from search
        articles = search_articles(query, start=start, count=batch_size)

        if articles:
            print(f"✅ Retrieved {len(articles)} articles in this batch")
            all_articles.extend(articles)
        else:
            print("⚠️ No articles returned in this batch. Stopping search.")
            break

    print(f"\n📚 Total articles collected: {len(all_articles)}")

    if all_articles:
        # Optional: Enrich articles with more details if you want better data
       # print("\n🔍 Enriching article details...")
        enriched_articles = enrich_article_details(all_articles)

        # Classify articles into barrier categories
       ## print("\n🧠 Classifying articles into barrier categories...")
        classified_articles = classify_articles_nlp(enriched_articles, barrier_examples)

        # Print sample results
        #print("\n📊 Sample of classified articles:")
        #for i, article in enumerate(classified_articles[:3]):
          #  print(f"{i + 1}. {article['Title']} - {article['Category']} ({article['Year']})")
          #  print(f"   Author(s): {article['Author(s)']}")
          #  print(f"   Link: {article['Link']}")
          #  print("-" * 60)

        # Create DataFrame
        df = pd.DataFrame(classified_articles)[["Author(s)", "Title", "Gaps", "Category", "Year", "Link"]]

        # Insert into Snowflake
        try:
            insert_to_snowflake(df, SNOWFLAKE_TABLE)
        except Exception as e:
            print(f"⚠️ Snowflake insertion skipped: {e}")
            print("✅ Data processing complete but not inserted to Snowflake.")

        return df
    else:
        print("❌ No articles were collected.")
        return pd.DataFrame()


# -------------------- EXECUTION --------------------
if __name__ == "__main__":
    print("🚀 Starting Elsevier article search and classification process")
    print("=" * 70)

    # Use a query focused on AI adoption barriers in healthcare
    query = '("artificial intelligence" OR "AI") AND ("primary care" OR "healthcare") AND ("adoption" OR "implementation" OR "barriers")'

    # Run the search and classification
    df = run_batched_search(query, total_to_fetch=50, batch_size=25)

    if not df.empty:
        print("\n✅ Process completed successfully")
        print(f"📊 Total articles processed: {len(df)}")

        # Save to CSV for backup/review
        try:
            csv_filename = "healthcare_ai_barriers.csv"
            df.to_csv(csv_filename, index=False)
            print(f"✅ Results saved to {csv_filename}")
        except Exception as e:
            print(f"⚠️ Could not save to CSV: {e}")
    else:
        print("\n❌ Process failed to collect any articles")