import snowflake.connector
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from transformers import pipeline
import os
import datetime
from sklearn.cluster import KMeans
from scipy.stats import f_oneway

# Define output directory
OUTPUT_DIR = r"C:\Users\Abebe\OneDrive\Desktop\BMIS690 Integrated Capstone\final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Timestamp for file names to avoid overwriting
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# Function to create a path for saving files
def get_output_path(filename):
    return os.path.join(OUTPUT_DIR, f"{timestamp}_{filename}")


# Snowflake connection parameters
SNOWFLAKE_USER = os.environ.get('SNOWFLAKE_USER', 'abebe')
SNOWFLAKE_PASSWORD = os.environ.get('SNOWFLAKE_PASSWORD', '$lo*Xq&z}2fQh:ja')
SNOWFLAKE_ACCOUNT = os.environ.get('SNOWFLAKE_ACCOUNT', 'sweqfnm-lx34353')
SNOWFLAKE_WAREHOUSE = os.environ.get('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH')
SNOWFLAKE_DATABASE = os.environ.get('SNOWFLAKE_DATABASE', 'STAGE_DB')
SNOWFLAKE_SCHEMA = os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')

# Create SQLAlchemy engine using snowflake.sqlalchemy.URL
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine

engine = create_engine(URL(
    user=SNOWFLAKE_USER,
    password=SNOWFLAKE_PASSWORD,
    account=SNOWFLAKE_ACCOUNT,
    warehouse=SNOWFLAKE_WAREHOUSE,
    database=SNOWFLAKE_DATABASE,
    schema=SNOWFLAKE_SCHEMA
))

# Fetch data from Snowflake tables using SQLAlchemy engine
elsevier_query = "SELECT * FROM ELSEVIER_ARTICLES_STAGING"
pubmed_query = "SELECT * FROM PUBMED_ARTICLES_STAGING"

# Load data into pandas DataFrames
elsevier_df = pd.read_sql(elsevier_query, engine)
pubmed_df = pd.read_sql(pubmed_query, engine)

# Combine datasets
df = pd.concat([elsevier_df, pubmed_df], ignore_index=True)

# Preprocessing - Check if columns exist before accessing
if 'category' in df.columns:
    df['category'] = df['category'].fillna('Unknown')
else:
    print("Warning: category column not found")

if 'gaps' in df.columns:
    df['gaps'] = df['gaps'].fillna('')
else:
    print("Warning: gaps column not found")

if 'year' in df.columns:
    df['year'] = df['year'].fillna(0).astype(int)
else:
    print("Warning: year column not found")

# Sentiment Analysis using BERT
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
df['sentiment_score'] = df['gaps'].apply(
    lambda x: sentiment_analyzer(x)[0]['label'] if isinstance(x, str) else "NEUTRAL")

# Topic Modeling with LDA
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(df['gaps'])

# LDA Model
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)
topic_values = lda.transform(X)
df['topic'] = topic_values.argmax(axis=1)

# K-Means Clustering for Topic Sentiment
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['topic']])

# Visualizing Sentiment by Topic
plt.figure(figsize=(10, 6))
topic_sentiment = df.groupby('topic')['sentiment_score'].value_counts().unstack().fillna(0)
topic_sentiment.plot(kind='bar', stacked=True)
plt.title('Sentiment Distribution by Topic')
plt.xlabel('Topic')
plt.ylabel('Sentiment Count')
plt.savefig(get_output_path('topic_sentiment_distribution.png'))
plt.close()

# Statistical Test: ANOVA to check differences in sentiment between topics
topic_groups = [df[df['topic'] == i]['sentiment_score'] for i in range(5)]
f_statistic, p_value = f_oneway(*topic_groups)
print(f"ANOVA Test Result: F-statistic={f_statistic}, p-value={p_value}")

# Classification Model to Predict Sentiment Based on Topics
df['sentiment_label'] = df['sentiment_score'].apply(lambda x: 1 if x == 'POSITIVE' else (0 if x == 'NEUTRAL' else -1))

X_train, X_test, y_train, y_test = train_test_split(df[['topic']], df['sentiment_label'], test_size=0.2,
                                                    random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save Classification Report
report_output_path = get_output_path('classification_report.txt')
with open(report_output_path, 'w') as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(report)

# Focus Areas Analysis
focus_areas = df.groupby('topic').agg({
    'sentiment_label': ['mean', 'std'],
    'category': 'count'
}).sort_values(('sentiment_label', 'mean'))

# Save Focus Areas Analysis
focus_areas_path = get_output_path('focus_areas_analysis.csv')
focus_areas.to_csv(focus_areas_path)

# Summary Report
summary_path = get_output_path('analysis_summary.txt')
with open(summary_path, 'w') as f:
    f.write("ANALYSIS SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Total Records: {len(df)}\n")
    f.write(f"Categories: {df['category'].nunique()}\n")
    f.write(f"Top Categories: {', '.join(df['category'].value_counts().nlargest(5).index.tolist())}\n")
    f.write(f"Sentiment Counts: {df['sentiment_score'].value_counts().to_dict()}\n")
    f.write("\nAnalysis Files:\n")
    f.write(f"Processed Data: {timestamp}_processed_data.csv\n")
    f.write(f"Sentiment Chart: {timestamp}_sentiment_distribution.png\n")
    f.write(f"Topic Sentiment Chart: {timestamp}_topic_sentiment.png\n")
    f.write(f"Classification Report: {timestamp}_classification_report.txt\n")
    f.write(f"Focus Areas Analysis: {timestamp}_focus_areas_analysis.csv\n")

# Close the Snowflake connection
conn.close()

print("\nAnalysis complete!")
