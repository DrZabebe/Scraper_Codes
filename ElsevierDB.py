
# -------------------- IMPORTS --------------------
import requests
import json
import time
import pandas as pd
import snowflake.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dotenv import load_dotenv
import os
from tenacity import retry, stop_after_attempt, wait_exponential


# -------------------- CONFIGURATION --------------------
#-------- Elsevier API Key
ELSEVIER_API_KEY = "" # If you have one
ELSEVIER_INSTTOKEN = ""

# Snowflake credentials
SNOWFLAKE_USER = ''
SNOWFLAKE_PASSWORD = ''
SNOWFLAKE_ACCOUNT = ''
SNOWFLAKE_WAREHOUSE = ''
SNOWFLAKE_DATABASE = ''
SNOWFLAKE_SCHEMA = ''
SNOWFLAKE_TABLE = 'elsevier_articles_staging'



# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# API Credentials and Configuration
ELSEVIER_API_KEY = os.getenv('ELSEVIER_API_KEY')
ELSEVIER_INSTTOKEN = os.getenv('ELSEVIER_INSTTOKEN')
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER')
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE')
SNOWFLAKE_SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')
SNOWFLAKE_TABLE = os.getenv('SNOWFLAKE_TABLE', 'AI_HEALTHCARE_BARRIERS')

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

def snowflake_connect():
    """
    Establishes a connection to Snowflake database.

    Returns:
        Connection object to Snowflake
    """
    try:
        return snowflake.connector.connect(
            account=SNOWFLAKE_ACCOUNT,
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA
        )
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {e}")
        raise


def insert_to_snowflake(df, table_name):
    """
    Insert DataFrame records into Snowflake table.

    Args:
        df: Pandas DataFrame containing article data
        table_name: Name of the Snowflake table
    """
    if df.empty:
        logger.warning("Empty DataFrame, nothing to insert into Snowflake")
        return

    try:
        conn = snowflake_connect()
        cs = conn.cursor()
        logger.info(f"Connected to Snowflake. Uploading {len(df)} records...")

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
        logger.info(f"Upload complete into {table_name}.")

    except snowflake.connector.errors.ProgrammingError as e:
        logger.error(f"Snowflake SQL error: {e}")
    except Exception as e:
        logger.error(f"Snowflake upload error: {e}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_articles_get_dois(query, start=0, count=100):
    """
    Search for articles using Elsevier Scopus API and return DOIs.

    Args:
        query: Search query string
        start: Starting index for pagination
        count: Number of results to return

    Returns:
        List of DOIs for matching articles
    """
    url = "https://api.elsevier.com/content/search/scopus"
    headers = {
        'X-ELS-APIKey': ELSEVIER_API_KEY,
        'X-ELS-Insttoken': ELSEVIER_INSTTOKEN,
        'Accept': 'application/json'
    }
    params = {
        'query': query,
        'count': count,
        'start': start,
        'date': '2000-2024',
        'view': 'COMPLETE'
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()  # Raise exception for non-200 status codes

        data = response.json()

        # Debug information
        results_count = data.get('search-results', {}).get('opensearch:totalResults', '0')
        logger.info(f"Found {results_count} total results")

        entries = data.get('search-results', {}).get('entry', [])
        dois = []

        for entry in entries:
            doi = entry.get('prism:doi')
            if doi:
                dois.append(doi)

        logger.info(f"Retrieved {len(dois)} DOIs in this batch")
        return dois

    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching articles: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response: {e}")
        logger.debug(response.text[:200] + "..." if len(response.text) > 200 else response.text)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search: {e}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_article_by_doi(doi):
    """
    Fetch detailed article metadata by DOI using Elsevier API.

    Args:
        doi: Digital Object Identifier for the article

    Returns:
        Dictionary containing article metadata or None if not found
    """
    # First try the full-text API
    url = f"https://api.elsevier.com/content/article/doi/{doi}"
    headers = {
        'X-ELS-APIKey': ELSEVIER_API_KEY,
        'X-ELS-Insttoken': ELSEVIER_INSTTOKEN,
        'Accept': 'application/json'
    }
    params = {
        'view': 'META_ABS'
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)

        # If full-text API fails, try abstract API
        if response.status_code != 200:
            logger.warning(f"Full-text API failed for DOI {doi}, trying abstract API...")
            url = f"https://api.elsevier.com/content/abstract/doi/{doi}"
            response = requests.get(url, headers=headers, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()

            # Handle different response structures
            if 'full-text-retrieval-response' in data:
                coredata = data['full-text-retrieval-response']['coredata']
                links = data['full-text-retrieval-response'].get('link', [])
                real_link = next((link.get('@href') for link in links if link.get('@ref') == 'scidir'),
                                 "No Link Available")
            elif 'abstracts-retrieval-response' in data:
                coredata = data['abstracts-retrieval-response']['coredata']
                links = data['abstracts-retrieval-response'].get('link', [])
                real_link = next((link.get('@href') for link in links if link.get('@ref') == 'scopus'),
                                 "No Link Available")
            else:
                logger.warning(f"Unexpected response structure for DOI {doi}")
                return None

            title = coredata.get('dc:title', 'No Title')

            # Handle different author formats
            authors = "Unknown Author"
            if 'dc:creator' in coredata:
                creator = coredata['dc:creator']
                if isinstance(creator, list):
                    authors = "; ".join([a.get('$', 'Unknown') for a in creator])
                elif isinstance(creator, dict):
                    authors = creator.get('$', 'Unknown Author')
                else:
                    authors = str(creator)
            elif 'authors' in data.get('abstracts-retrieval-response', {}):
                author_list = data['abstracts-retrieval-response']['authors'].get('author', [])
                if isinstance(author_list, list):
                    author_names = []
                    for author in author_list:
                        if 'preferred-name' in author:
                            surname = author['preferred-name'].get('surname', '')
                            given_name = author['preferred-name'].get('given-name', '')
                            author_names.append(f"{surname}, {given_name}")
                    if author_names:
                        authors = "; ".join(author_names)

            # Get abstract
            abstract = coredata.get('dc:description', '')

            # Get publication year
            pub_date = coredata.get('prism:coverDate', '')
            pub_year = pub_date[:4] if pub_date else 'Unknown'

            return {
                "Author(s)": authors,
                "Title": title,
                "Text": f"{title} {abstract}",
                "Year": pub_year,
                "Link": real_link,
                "DOI": doi
            }

        elif response.status_code == 404:
            logger.warning(f"DOI not found (404): {doi}")
            return None
        else:
            logger.error(f"Error fetching DOI {doi}: {response.status_code}")
            logger.debug(response.text[:200] + "..." if len(response.text) > 200 else response.text)
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for DOI {doi}: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response for DOI {doi}: {e}")
        logger.debug(response.text[:200] + "..." if len(response.text) > 200 else response.text)
        raise
    except Exception as e:
        logger.error(f"Unexpected error for DOI {doi}: {e}")
        return None


def classify_articles_nlp(articles, barrier_examples):
    """
    Classify articles into predefined barrier categories using NLP.

    Args:
        articles: List of article dictionaries
        barrier_examples: Dictionary mapping category names to example texts

    Returns:
        List of articles with added category classifications
    """
    if not articles:
        logger.warning("No articles to classify")
        return []

    barrier_names = list(barrier_examples.keys())
    barrier_texts = list(barrier_examples.values())

    all_texts = [article["Text"] for article in articles] + barrier_texts

    # Use TF-IDF vectorization for text comparison
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    article_vectors = tfidf_matrix[:-len(barrier_texts)]
    category_vectors = tfidf_matrix[-len(barrier_texts):]

    # Calculate cosine similarity between articles and categories
    similarities = cosine_similarity(article_vectors, category_vectors)
    categories = [barrier_names[i.argmax()] for i in similarities]

    # Add classifications to articles and clean up
    for idx, article in enumerate(articles):
        article["Category"] = categories[idx]
        # Truncate text to use as "gaps" summary
        article["Gaps"] = article["Text"][:400]
        del article["Text"]  # Remove original text to save space

    return articles


def run_batched(query, total_to_fetch=500, batch_size=100):
    """
    Main function to run the article search, retrieval, and classification process in batches.

    Args:
        query: Search query string
        total_to_fetch: Total number of articles to fetch
        batch_size: Size of each batch for pagination

    Returns:
        Pandas DataFrame with classified articles
    """
    all_articles = []

    try:
        for start in range(0, total_to_fetch, batch_size):
            logger.info(f"Fetching batch {start + 1} to {min(start + batch_size, total_to_fetch)}")

            # Add rate limiting to avoid API rate limits
            if start > 0:
                logger.info("Waiting 3 seconds before next batch...")
                time.sleep(3)

            dois = search_articles_get_dois(query, start=start, count=batch_size)
            if not dois:
                logger.warning("No DOIs returned. Moving to processing collected articles.")
                break

            batch_articles = []
            for i, doi in enumerate(dois):
                logger.info(f"Fetching DOI {i + 1}/{len(dois)}: {doi}")
                article = fetch_article_by_doi(doi)
                if article:
                    batch_articles.append(article)

                # Add small delay between requests
                if i < len(dois) - 1:
                    time.sleep(0.75)  # Increased delay to reduce rate limiting issues

            if batch_articles:
                all_articles.extend(batch_articles)
                logger.info(f"Collected {len(batch_articles)} articles in this batch")
            else:
                logger.warning("No articles processed in this batch.")

        if all_articles:
            logger.info(f"Classifying {len(all_articles)} articles...")
            classified_articles = classify_articles_nlp(all_articles, barrier_examples)
            df = pd.DataFrame(classified_articles)[["Author(s)", "Title", "Gaps", "Category", "Year", "Link"]]

            # Save to CSV as backup before Snowflake insertion
            df.to_csv("ai_healthcare_barriers.csv", index=False)
            logger.info("Saved backup to CSV file")

            insert_to_snowflake(df, SNOWFLAKE_TABLE)
            return df
        else:
            logger.error("No articles were collected across all batches.")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error in run_batched: {e}")
        # Save any collected articles to avoid complete data loss
        if all_articles:
            try:
                pd.DataFrame(all_articles).to_csv("ai_healthcare_barriers_partial.csv", index=False)
                logger.info("Saved partial results to CSV due to error")
            except:
                pass
        return pd.DataFrame()


def test_api_connection():
    """
    Test the Elsevier API connection without Snowflake.

    Returns:
        Boolean indicating if connection was successful
    """
    logger.info("Testing Elsevier API connection...")
    query = '("artificial intelligence" OR "AI") AND ("healthcare")'

    try:
        url = "https://api.elsevier.com/content/search/scopus"
        headers = {
            'X-ELS-APIKey': ELSEVIER_API_KEY,
            'X-ELS-Insttoken': ELSEVIER_INSTTOKEN,
            'Accept': 'application/json'
        }
        params = {
            'query': query,
            'count': 1,
            'view': 'COMPLETE'
        }

        response = requests.get(url, headers=headers, params=params, timeout=30)
        logger.info(f"Status code: {response.status_code}")

        if response.status_code == 200:
            logger.info("API connection successful")
            data = response.json()
            total_results = data.get('search-results', {}).get('opensearch:totalResults', '0')
            logger.info(f"Total results available: {total_results}")
            return True
        else:
            logger.error(f"API connection failed: {response.status_code}")
            logger.debug(response.text[:200] + "..." if len(response.text) > 200 else response.text)
            return False

    except Exception as e:
        logger.error(f"Error testing API connection: {e}")
        return False


# -------------------- EXECUTION --------------------
if __name__ == "__main__":
    # First test API connection
    if test_api_connection():
        # Modify the query to get better results
        query = '''
            ("artificial intelligence" OR "AI" OR "machine learning") 
            AND ("primary care" OR "healthcare" OR "clinical practice") 
            AND ("adoption" OR "implementation" OR "barriers" OR "challenges" OR "limitations")
        '''

        # Run with smaller batch for testing, increase for production
        df = run_batched(query, total_to_fetch=50, batch_size=10)

        if not df.empty:
            logger.info(f"Successfully processed {len(df)} articles")
            logger.info("\nSample data:")
            print(df[["Title", "Category", "Year"]].head())

            # Analyze category distribution
            category_counts = df["Category"].value_counts()
            logger.info("\nCategory distribution:")
            print(category_counts)
    else:
        logger.error("Please check your Elsevier API key and connection settings.")