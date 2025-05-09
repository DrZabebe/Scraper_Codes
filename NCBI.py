
from Bio import Entrez, Medline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from azure.storage.filedatalake import DataLakeServiceClient
import pandas as pd



# Azure credentials
STORAGE_ACCOUNT_NAME = 'pubmedncbi'
STORAGE_ACCOUNT_KEY = 'yourstorageaccountkey'
FILE_SYSTEM_NAME = 'pubmedncbi'
DIRECTORY_NAME = 'ai-barriers'
PARQUET_FILE_NAME = 'ai_pubmed_nlp_categorized_with_year.parquet'



# Set your email for Entrez (required)
Entrez.email = "gzne.2021@gmail.com"  # Replace with your email

# Define categories and example definitions
barrier_examples = {
    "Trust and Lack of Transparency": "AI systems are often opaque, with black-box models that limit transparency and reduce trust among clinicians.",
    "Loss of Autonomy and Clinical Authority": "AI systems may override or influence clinical decisions, causing concerns about loss of control and autonomy.",
    "Training Deficits and Cognitive Burden": "Clinicians often lack formal AI training, which leads to cognitive overload and low confidence in system usage.",
    "System Design Failures and Workflow Disruption": "Poorly integrated AI tools disrupt established workflows and reduce usability in primary care settings.",
    "Organizational Readiness and Structural Constraints": "Many healthcare settings lack the infrastructure, leadership support, or readiness to adopt AI effectively.",
    "Socio-Cultural and Patient Interaction Barriers": "Cultural differences, patient trust, and fear of dehumanized care affect AI adoption in clinical environments."
}

# Search PubMed for relevant articles
def search_pubmed(query, max_results=20):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    return record["IdList"]

# Fetch details using PubMed IDs
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
        pub_year = record.get("DP", "Unknown")[:4]  # Extract year only
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

# Classify articles using cosine similarity
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

# Main execution
query = "AI adoption in primary care barriers"
ids = search_pubmed(query, max_results=25)
raw_articles = fetch_details(ids)
classified_articles = classify_articles_nlp(raw_articles, barrier_examples)

# Save results
df = pd.DataFrame(classified_articles)
df.to_csv(r'C:\Users\Abebe\Downloads\ai_barriers_categorized3.csv', index=False)
