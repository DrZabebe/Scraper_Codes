import pandas as pd
import pytz
from datetime import datetime
import requests  # For Elsevier API calls
import snowflake.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- CONFIG --------------------

# Snowflake credentials
SNOWFLAKE_USER = 'abebe'
SNOWFLAKE_PASSWORD = '$lo*Xq&z}2fQh:ja'
SNOWFLAKE_ACCOUNT = 'sweqfnm-lx34353'
SNOWFLAKE_WAREHOUSE = 'L_WAREHOUSE'
SNOWFLAKE_DATABASE = 'STAGE_DB'
SNOWFLAKE_SCHEMA = 'PUBLIC'
SNOWFLAKE_TABLE = 'elsevier_articles_staging'

TODAY = datetime.today().strftime("%Y-%m-%d")

barrier_examples = {
    "Trust and Clinical Autonomy": "Physicians express reluctance to use AI due to familiarity with patient conditions and value of clinical intuition in decision-making.",
    "Human-AI Interaction": "The lack of human intervention in AI systems is a barrier for physicians who value the patient-provider relationship.",
    "Workflow Integration": "Poor AI integration disrupts clinical workflows and reduces usability in primary care settings.",
    "Technical Infrastructure": "Many primary care practices lack the infrastructure and technical resources to implement AI solutions.",
    "Data Quality and Privacy": "Incomplete medical data and privacy concerns undermine AI reliability and patient trust.",
    "Clinical Expertise and Training": "Clinicians lack AI training, causing cognitive overload and low confidence in using AI tools.",
    "Regulatory and Compliance Issues": "Privacy regulations and approval processes delay AI implementation in healthcare settings."
}

# Global tracking of processed authors
processed_authors = set()


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


def remove_duplicates(df):
    # First, keep track of authors we've already seen globally
    global processed_authors

    # Create a copy to avoid modifying the original during iteration
    df_cleaned = df.copy()

    # Create a mask for rows to keep
    rows_to_keep = []

    for index, row in df.iterrows():
        authors = row['Author(s)']

        # If we've already processed this author before, skip this row
        if authors in processed_authors:
            rows_to_keep.append(False)
        else:
            # Add this author to our global tracking set
            processed_authors.add(authors)
            rows_to_keep.append(True)

    # Apply the mask to keep only unique author entries
    return df_cleaned[rows_to_keep]


def insert_to_snowflake(df, table_name):
    try:
        conn = snowflake_connect()
        cs = conn.cursor()
        print(f"✅ Connected to Snowflake. Uploading {len(df)} records...")

        # Get the current time in Central Time (CT)
        current_time_utc = datetime.now(pytz.utc)
        central_time = current_time_utc.astimezone(pytz.timezone('America/Chicago'))

        # Convert to string format for Snowflake insertion
        central_time_str = central_time.strftime('%Y-%m-%d %H:%M:%S')

        # Clean duplicates before inserting
        df_cleaned = remove_duplicates(df)

        print(f"After removing duplicates, uploading {len(df_cleaned)} records...")

        # Create the insert SQL with the correct timestamp for CREATED_AT
        for index, row in df_cleaned.iterrows():
            sql = f"""
            INSERT INTO {table_name} ("AUTHOR", "TITLE", "GAPS", "CATEGORY", "YEAR", "LINK", "CREATED_AT")
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cs.execute(sql, (
                row["Author(s)"],
                row["Title"],
                row["Gaps"],
                row["Category"],
                row["Year"],
                row["Link"],
                central_time_str  # Pass the Central Time as CREATED_AT
            ))

        cs.close()
        conn.close()
        print(f"✅ Upload complete into {table_name}.\n")

    except Exception as e:
        print(f"❌ Snowflake Upload Error: {e}")


def search_elsevier(query, total_results=500, count=1):
    url = "https://api.elsevier.com/content/search/scopus"
    headers = {
        'X-ELS-APIKey': '0228f90ebd24ad17a13e788b4d803079',  # Replace with your actual API key
        'X-ELS-Insttoken': 'f52020ea6e9786abb97bf7d9f7155fff',  # Replace with your actual insttoken
        'Accept': 'application/json'
    }
    params = {
        'query': query,
        'count': count,
        'view': 'COMPLETE'
    }
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        print(f"Total Results: {data['search-results']['opensearch:totalResults']}")
        return data
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def fetch_elsevier_details(data):
    articles = []
    if 'search-results' in data:
        for entry in data['search-results']['entry']:
            title = entry.get('dc:title', 'No Title')
            authors = "; ".join([author['authname'] for author in entry.get('author', [])])
            description = entry.get('dc:description', 'No description available')
            year = entry.get('prism:coverDate', 'Unknown')[:4]

            # Filter for articles only from 2019-2025
            if not (2019 <= int(year) <= 2025):
                continue

            doi = entry.get('prism:doi', None)
            # Create the DOI link if DOI exists
            if doi:
                doi_link = f"https://doi.org/{doi}"
            else:
                doi_link = 'No DOI link'

            articles.append({
                "Author(s)": authors,
                "Title": title,
                "Gaps": description[:400],  # First 400 characters of the description as 'gaps'
                "Category": 'AI in Healthcare',  # Example category
                "Year": year,
                "Link": doi_link  # Use the DOI link here
            })
    return articles


def classify_articles_nlp(articles, barrier_examples):
    barrier_names = list(barrier_examples.keys())
    barrier_texts = list(barrier_examples.values())

    all_texts = [article["Gaps"] for article in articles] + barrier_texts
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    article_vectors = tfidf_matrix[:-len(barrier_texts)]
    category_vectors = tfidf_matrix[-len(barrier_texts):]

    similarities = cosine_similarity(article_vectors, category_vectors)
    categories = [barrier_names[i.argmax()] for i in similarities]

    for idx, article in enumerate(articles):
        article["Category"] = categories[idx]
        article["Gaps"] = article["Gaps"]
    return articles


# -------------------- IMPROVED API FETCHING --------------------
def search_elsevier_with_pagination(query, max_results=1000):
    """Fetch as many results as possible from Elsevier using pagination"""
    url = "https://api.elsevier.com/content/search/scopus"
    headers = {
        'X-ELS-APIKey': '0228f90ebd24ad17a13e788b4d803079',
        'X-ELS-Insttoken': 'f52020ea6e9786abb97bf7d9f7155fff',
        'Accept': 'application/json'
    }

    all_articles = []
    results_per_page = 25  # Elsevier's API typically allows 25-100 results per page
    current_page = 0

    print(f"Starting to collect data for query: {query}")

    while True:
        # Calculate start position based on current page
        start = current_page * results_per_page

        # If we've reached our maximum desired results, break
        if start >= max_results:
            print(f"Reached maximum result limit of {max_results}")
            break

        params = {
            'query': query,
            'count': results_per_page,
            'start': start,
            'view': 'COMPLETE',
            'date': '2019-2025',  # Add date range filter directly in the API call
            'sort': 'relevancy'
        }

        print(f"Fetching page {current_page + 1} (records {start + 1}-{start + results_per_page})")

        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()

        # Get the actual results for this page
        results = data.get('search-results', {}).get('entry', [])
        total_results = int(data.get('search-results', {}).get('opensearch:totalResults', 0))

        print(f"Retrieved {len(results)} results (total available: {total_results})")

        if not results:
            print("No more results to fetch")
            break

        # Process the current page results
        page_articles = fetch_elsevier_details({'search-results': {'entry': results}})
        all_articles.extend(page_articles)

        # If we've fetched all available results, break
        if (start + len(results)) >= total_results:
            print("Fetched all available results")
            break

        # Move to next page
        current_page += 1

    return all_articles


# List of topics to search for
topics = [
    '("artificial intelligence" OR "AI") AND ("primary care" OR "healthcare") AND ("adoption" OR "implementation")',
    '("machine learning" OR "deep learning") AND ("healthcare" OR "medicine") AND ("applications" OR "challenges")',
    '("clinical decision support" OR "AI healthcare") AND ("barriers" OR "challenges")',
    '("healthcare systems" OR "AI implementation") AND ("efficiency" OR "cost reduction")',
    '("AI ethics" OR "machine learning transparency") AND ("healthcare")'
]


# -------------------- MAIN EXECUTION --------------------
def main():
    # Reset the processed_authors set before starting
    global processed_authors
    processed_authors = set()

    all_processed_articles = []

    # Process each topic
    for topic in topics:
        print(f"\n🔍 Processing topic: {topic}")

        # Get all articles for this topic
        articles = search_elsevier_with_pagination(topic, max_results=1000)

        if articles:
            # Classify the articles
            classified = classify_articles_nlp(articles, barrier_examples)

            # Convert to DataFrame
            df = pd.DataFrame(classified)[["Author(s)", "Title", "Gaps", "Category", "Year", "Link"]]

            # Insert to Snowflake (which will handle de-duplication)
            insert_to_snowflake(df, SNOWFLAKE_TABLE)

            # Keep track of all processed articles
            all_processed_articles.extend(classified)

            print(f"✅ Processed {len(articles)} articles for topic '{topic}'")
        else:
            print(f"⚠️ No articles found for topic '{topic}'")

    print(f"\n✅ COMPLETED: Processed {len(all_processed_articles)} total articles across all topics")
    print(f"✅ After deduplication, {len(processed_authors)} unique authors were included")


# Run the main function
# Execute the main function
main()