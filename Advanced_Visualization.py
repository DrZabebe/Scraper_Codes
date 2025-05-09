import snowflake.connector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_curve
from transformers import pipeline
import os
import datetime
from sklearn.cluster import KMeans
from scipy.stats import f_oneway
from wordcloud import WordCloud
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Define output directory
OUTPUT_DIR = r""
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Timestamp for file names to avoid overwriting
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# Function to create a path for saving files
def get_output_path(filename):
    return os.path.join(OUTPUT_DIR, f"{timestamp}_{filename}")


# Set plot style for better visualizations
plt.style.use('ggplot')
sns.set(style="whitegrid")


# Function to handle errors and provide informative messages
def safe_execute(func, error_message, default_return=None):
    try:
        return func()
    except Exception as e:
        print(f"{error_message}: {str(e)}")
        return default_return


# Snowflake connection parameters
SNOWFLAKE_USER = os.environ.get('SNOWFLAKE_USER', '')
SNOWFLAKE_PASSWORD = os.environ.get('SNOWFLAKE_PASSWORD', '')
SNOWFLAKE_ACCOUNT = os.environ.get('SNOWFLAKE_ACCOUNT', '')
SNOWFLAKE_WAREHOUSE = os.environ.get('SNOWFLAKE_WAREHOUSE', '')
SNOWFLAKE_DATABASE = os.environ.get('SNOWFLAKE_DATABASE', '')
SNOWFLAKE_SCHEMA = os.environ.get('SNOWFLAKE_SCHEMA', '')

print("Starting data analysis...")
print(f"Results will be saved to: {OUTPUT_DIR}")

try:
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

    # Create direct connection for cursor operations
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )

    print("Successfully connected to Snowflake")

    # Fetch data from Snowflake tables using SQLAlchemy engine
    elsevier_query = "SELECT * FROM ELSEVIER_ARTICLES_STAGING"
    pubmed_query = "SELECT * FROM PUBMED_ARTICLES_STAGING"

    # Load data into pandas DataFrames
    print("Retrieving data from Snowflake...")
    elsevier_df = pd.read_sql(elsevier_query, engine)
    pubmed_df = pd.read_sql(pubmed_query, engine)

    # Print basic information about the datasets
    print(f"Elsevier data: {elsevier_df.shape[0]} rows, {elsevier_df.shape[1]} columns")
    print(f"PubMed data: {pubmed_df.shape[0]} rows, {pubmed_df.shape[1]} columns")

    # Combine datasets
    df = pd.concat([elsevier_df, pubmed_df], ignore_index=True)
    print(f"Combined dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Save the raw combined data
    df.to_csv(get_output_path('raw_combined_data.csv'), index=False)

    # Preprocessing - Check if columns exist before accessing
    print("Preprocessing data...")

    # Convert column names to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]

    if 'category' in df.columns:
        df['category'] = df['category'].fillna('Unknown')
        category_counts = df['category'].value_counts()
        print(f"Top 5 categories: {', '.join(category_counts.nlargest(5).index.tolist())}")
    else:
        print("Warning: category column not found")
        df['category'] = 'Unknown'

    if 'gaps' in df.columns:
        df['gaps'] = df['gaps'].fillna('')
        print(
            f"Gaps text statistics: min length={df['gaps'].str.len().min()}, max length={df['gaps'].str.len().max()}, average length={df['gaps'].str.len().mean():.1f}")
    else:
        print("Warning: gaps column not found")
        df['gaps'] = ''

    if 'year' in df.columns:
        df['year'] = df['year'].fillna(0).astype(int)
        year_range = f"{df['year'].min()} - {df['year'].max()}"
        print(f"Year range: {year_range}")
    else:
        print("Warning: year column not found")
        df['year'] = datetime.datetime.now().year

    # Create word count feature
    df['word_count'] = df['gaps'].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)

    # Remove rows with empty gaps text for analysis
    analysis_df = df[df['gaps'].str.len() > 10].copy()
    print(f"Filtered dataset for analysis: {analysis_df.shape[0]} rows with meaningful text")

    # Save the preprocessed data
    df.to_csv(get_output_path('preprocessed_data.csv'), index=False)

    # Advanced Text Processing and Analysis
    print("Performing advanced text analysis...")

    # Sentiment Analysis using BERT
    print("Running sentiment analysis...")
    sentiment_analyzer = pipeline("sentiment-analysis",
                                  model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

    # Process sentiment in batches to avoid memory issues
    batch_size = 100
    sentiments = []

    for i in range(0, len(analysis_df), batch_size):
        batch = analysis_df['gaps'].iloc[i:i + batch_size].tolist()
        batch_sentiments = []

        for text in batch:
            if not isinstance(text, str) or len(text.strip()) < 5:
                batch_sentiments.append("NEUTRAL")
            else:
                try:
                    result = sentiment_analyzer(text[:512])[0]  # Truncate to avoid token limits
                    batch_sentiments.append(result['label'])
                except Exception as e:
                    print(f"Error in sentiment analysis: {e}")
                    batch_sentiments.append("NEUTRAL")

        sentiments.extend(batch_sentiments)
        print(f"Processed sentiment for {min(i + batch_size, len(analysis_df))}/{len(analysis_df)} records")

    analysis_df['sentiment_score'] = sentiments

    # Convert sentiment to numeric for analysis
    analysis_df['sentiment_label'] = analysis_df['sentiment_score'].apply(
        lambda x: 1 if x == 'POSITIVE' else (-1 if x == 'NEGATIVE' else 0)
    )

    # Save sentiment distribution visualization
    plt.figure(figsize=(10, 6))
    sentiment_counts = analysis_df['sentiment_score'].value_counts()
    ax = sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
    plt.title('Sentiment Distribution', fontsize=16)
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Count', fontsize=14)

    # Add count labels on bars
    for i, count in enumerate(sentiment_counts):
        ax.text(i, count + 5, str(count), ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(get_output_path('sentiment_distribution.png'), dpi=300)
    plt.close()

    # Topic Modeling with LDA
    print("Performing topic modeling...")

    # Create document-term matrix
    count_vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=10000,
        stop_words='english'
    )

    doc_term_matrix = count_vectorizer.fit_transform(analysis_df['gaps'])
    feature_names = count_vectorizer.get_feature_names_out()

    # Optimize number of topics using perplexity
    perplexity_scores = []
    topic_range = range(2, 11)

    for n_topics in topic_range:
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_method='online',
            random_state=42,
            n_jobs=-1
        )
        lda_model.fit(doc_term_matrix)
        perplexity_scores.append(lda_model.perplexity(doc_term_matrix))
        print(f"Completed LDA model with {n_topics} topics")

    # Plot perplexity scores
    plt.figure(figsize=(10, 6))
    plt.plot(topic_range, perplexity_scores, marker='o')
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity Score')
    plt.title('Optimal Number of Topics')
    plt.grid(True)
    plt.savefig(get_output_path('topic_perplexity.png'), dpi=300)
    plt.close()

    # Find optimal number of topics
    optimal_topics = topic_range[perplexity_scores.index(min(perplexity_scores))] if perplexity_scores else 5
    print(f"Optimal number of topics: {optimal_topics}")

    # Train final LDA model with optimal number of topics
    lda_model = LatentDirichletAllocation(
        n_components=optimal_topics,
        max_iter=20,
        learning_method='online',
        random_state=42,
        n_jobs=-1
    )

    lda_model.fit(doc_term_matrix)

    # Get topic keywords for each topic
    topic_keywords = []
    topic_labels = []

    for topic_idx, topic in enumerate(lda_model.components_):
        top_keywords_idx = topic.argsort()[:-11:-1]  # Top 10 keywords
        top_keywords = [feature_names[i] for i in top_keywords_idx]
        topic_keywords.append(top_keywords)
        topic_labels.append(f"Topic {topic_idx + 1}: {', '.join(top_keywords[:3])}")

        # Create word cloud for each topic
        topic_dict = {feature_names[i]: topic[i] for i in top_keywords_idx}
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=50
        ).generate_from_frequencies(topic_dict)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {topic_idx + 1} Keywords', fontsize=16)
        plt.tight_layout()
        plt.savefig(get_output_path(f'topic_{topic_idx + 1}_wordcloud.png'), dpi=300)
        plt.close()

    # Save topic keywords to file
    with open(get_output_path('topic_keywords.txt'), 'w') as f:
        for topic_idx, keywords in enumerate(topic_keywords):
            f.write(f"Topic {topic_idx + 1}: {', '.join(keywords)}\n\n")

    # Transform documents to topic space
    doc_topic_distribution = lda_model.transform(doc_term_matrix)

    # Assign dominant topic to each document
    analysis_df['topic'] = doc_topic_distribution.argmax(axis=1)

    # Add topic distribution columns for each topic
    for i in range(optimal_topics):
        analysis_df[f'topic_{i + 1}_prob'] = doc_topic_distribution[:, i]

    # t-SNE visualization of topics
    print("Creating t-SNE visualization...")

    # Use t-SNE to visualize document clusters
    tsne_model = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    tsne_results = tsne_model.fit_transform(doc_topic_distribution)

    # Create a dataframe for visualization
    tsne_df = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'topic': analysis_df['topic'],
        'sentiment': analysis_df['sentiment_score']
    })

    # Plot t-SNE results colored by topic
    plt.figure(figsize=(12, 10))
    for topic_idx in range(optimal_topics):
        subset = tsne_df[tsne_df['topic'] == topic_idx]
        plt.scatter(subset['x'], subset['y'], label=f'Topic {topic_idx + 1}', alpha=0.7)

    plt.title('t-SNE Visualization of Document Topics', fontsize=16)
    plt.legend(title="Topics")
    plt.tight_layout()
    plt.savefig(get_output_path('tsne_topics.png'), dpi=300)
    plt.close()

    # Plot t-SNE results colored by sentiment
    plt.figure(figsize=(12, 10))
    colors = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'blue'}

    for sentiment, color in colors.items():
        subset = tsne_df[tsne_df['sentiment'] == sentiment]
        plt.scatter(subset['x'], subset['y'], c=color, label=sentiment, alpha=0.7)

    plt.title('t-SNE Visualization of Document Sentiments', fontsize=16)
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(get_output_path('tsne_sentiments.png'), dpi=300)
    plt.close()

    # Visualizing Sentiment by Topic
    print("Analyzing sentiment distribution by topic...")

    # Create a pivot table for visualization
    topic_sentiment = pd.crosstab(
        analysis_df['topic'],
        analysis_df['sentiment_score'],
        normalize='index'
    ) * 100  # Convert to percentage

    # Plot stacked bar chart
    plt.figure(figsize=(12, 8))
    topic_sentiment.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Sentiment Distribution by Topic (%)', fontsize=16)
    plt.xlabel('Topic', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.xticks(range(optimal_topics), [f"Topic {i + 1}" for i in range(optimal_topics)], rotation=0)
    plt.legend(title="Sentiment")

    # Add percentage labels
    for i, topic in enumerate(topic_sentiment.index):
        cumulative_sum = 0
        for sentiment in topic_sentiment.columns:
            value = topic_sentiment.loc[topic, sentiment]
            if value > 5:  # Only show labels for segments > 5%
                plt.text(i, cumulative_sum + value / 2, f"{value:.1f}%", ha='center', va='center')
            cumulative_sum += value

    plt.tight_layout()
    plt.savefig(get_output_path('topic_sentiment_distribution.png'), dpi=300)
    plt.close()

    # Statistical Test: ANOVA to check differences in sentiment between topics
    print("Performing statistical analysis...")

    # Prepare data for ANOVA
    topic_groups = [analysis_df[analysis_df['topic'] == i]['sentiment_label'] for i in range(optimal_topics)]
    topic_groups = [group for group in topic_groups if len(group) > 0]  # Remove empty groups

    if len(topic_groups) >= 2:  # Need at least 2 groups for ANOVA
        f_statistic, p_value = f_oneway(*topic_groups)
        anova_result = f"ANOVA Test Result: F-statistic={f_statistic:.4f}, p-value={p_value:.4f}"
        print(anova_result)

        # Interpret the result
        if p_value < 0.05:
            anova_interpretation = "There is a statistically significant difference in sentiment between topics (p < 0.05)."
        else:
            anova_interpretation = "There is no statistically significant difference in sentiment between topics (p >= 0.05)."

        print(anova_interpretation)
    else:
        anova_result = "Could not perform ANOVA test: need at least 2 non-empty topic groups"
        anova_interpretation = ""
        print(anova_result)

    # Classification Model to Predict Sentiment Based on Topics
    print("Training classification model...")

    # Prepare features for classification
    # Include topic probabilities and other relevant features
    feature_cols = [f'topic_{i + 1}_prob' for i in range(optimal_topics)] + ['word_count']

    # Make sure all feature columns exist in the dataframe
    for col in feature_cols:
        if col not in analysis_df.columns:
            analysis_df[col] = 0

    # Split data into training and testing sets
    X = analysis_df[feature_cols]
    y = analysis_df['sentiment_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier with hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best model parameters: {grid_search.best_params_}")

    # Evaluate model on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Model accuracy: {accuracy:.4f}")
    print(report)

    # Create confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(get_output_path('confusion_matrix.png'), dpi=300)
    plt.close()

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.savefig(get_output_path('feature_importance.png'), dpi=300)
    plt.close()

    # Save Classification Report
    report_output_path = get_output_path('classification_report.txt')
    with open(report_output_path, 'w') as f:
        f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
        f.write(f"Best Model Parameters: {grid_search.best_params_}\n\n")
        f.write(report)
        f.write("\n\n")
        f.write(anova_result + "\n")
        f.write(anova_interpretation + "\n")

    # Focus Areas Analysis
    print("Analyzing key focus areas...")

    focus_areas = analysis_df.groupby('topic').agg({
        'sentiment_label': ['mean', 'std', 'count'],
        'category': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown',
        'year': ['min', 'max']
    }).reset_index()

    # Rename columns for clarity
    focus_areas.columns = [
        'topic',
        'avg_sentiment',
        'std_sentiment',
        'document_count',
        'primary_category',
        'earliest_year',
        'latest_year'
    ]

    # Add topic keywords
    focus_areas['top_keywords'] = [', '.join(keywords[:5]) for keywords in topic_keywords]

    # Sort by average sentiment
    focus_areas = focus_areas.sort_values('avg_sentiment')

    # Save Focus Areas Analysis
    focus_areas_path = get_output_path('focus_areas_analysis.csv')
    focus_areas.to_csv(focus_areas_path, index=False)

    # Visualize focus areas
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    sns.barplot(x='topic', y='avg_sentiment', data=focus_areas, palette='RdYlGn')
    plt.title('Average Sentiment by Topic', fontsize=16)
    plt.xlabel('Topic', fontsize=14)
    plt.ylabel('Average Sentiment', fontsize=14)
    plt.xticks(range(len(focus_areas)), [f"Topic {i + 1}" for i in range(len(focus_areas))], rotation=0)

    plt.subplot(2, 1, 2)
    sns.barplot(x='topic', y='document_count', data=focus_areas, palette='Blues')
    plt.title('Document Count by Topic', fontsize=16)
    plt.xlabel('Topic', fontsize=14)
    plt.ylabel('Document Count', fontsize=14)
    plt.xticks(range(len(focus_areas)), [f"Topic {i + 1}" for i in range(len(focus_areas))], rotation=0)

    plt.tight_layout()
    plt.savefig(get_output_path('focus_areas_analysis.png'), dpi=300)
    plt.close()

    # Save processed data with all features
    analysis_df.to_csv(get_output_path('full_analysis_data.csv'), index=False)

    # Summary Report
    print("Creating summary report...")

    summary_path = get_output_path('analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("COMPREHENSIVE TEXT ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("DATA OVERVIEW\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Records Used for Analysis: {len(analysis_df)}\n")
        f.write(f"Categories: {df['category'].nunique()}\n")
        if 'category' in df.columns:
            f.write(f"Top Categories: {', '.join(df['category'].value_counts().nlargest(5).index.tolist())}\n")
        if 'year' in df.columns:
            f.write(f"Year Range: {df['year'].min()} - {df['year'].max()}\n")
        f.write("\n")

        f.write("SENTIMENT ANALYSIS\n")
        f.write("-" * 60 + "\n")
        sentiment_stats = analysis_df['sentiment_score'].value_counts()
        for sentiment, count in sentiment_stats.items():
            percentage = 100 * count / len(analysis_df)
            f.write(f"{sentiment}: {count} ({percentage:.1f}%)\n")
        f.write("\n")

        f.write("TOPIC MODELING\n")
        f.write("-" * 60 + "\n")
        f.write(f"Optimal Number of Topics: {optimal_topics}\n\n")

        for topic_idx, keywords in enumerate(topic_keywords):
            f.write(f"Topic {topic_idx + 1}: {', '.join(keywords[:10])}\n")
            topic_docs = len(analysis_df[analysis_df['topic'] == topic_idx])
            topic_percentage = 100 * topic_docs / len(analysis_df)
            f.write(f"Documents: {topic_docs} ({topic_percentage:.1f}%)\n\n")

        f.write("STATISTICAL ANALYSIS\n")
        f.write("-" * 60 + "\n")
        f.write(anova_result + "\n")
        f.write(anova_interpretation + "\n\n")

        f.write("CLASSIFICATION MODEL\n")
        f.write("-" * 60 + "\n")
        f.write(f"Model Accuracy: {accuracy:.4f}\n")
        f.write(f"Best Model Parameters: {grid_search.best_params_}\n\n")

        f.write("Top Feature Importance:\n")
        for _, row in feature_importance.head(5).iterrows():
            f.write(f"- {row['feature']}: {row['importance']:.4f}\n")
        f.write("\n")

        f.write("FOCUS AREAS\n")
        f.write("-" * 60 + "\n")
        for _, row in focus_areas.iterrows():
            f.write(f"Topic {row['topic'] + 1}: {row['top_keywords']}\n")
            f.write(f"  Primary Category: {row['primary_category']}\n")
            f.write(f"  Average Sentiment: {row['avg_sentiment']:.2f}\n")
            f.write(f"  Document Count: {row['document_count']}\n")
            f.write(f"  Year Range: {int(row['earliest_year'])} - {int(row['latest_year'])}\n\n")

        f.write("OUTPUT FILES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Raw Data: {timestamp}_raw_combined_data.csv\n")
        f.write(f"Preprocessed Data: {timestamp}_preprocessed_data.csv\n")
        f.write(f"Full Analysis Data: {timestamp}_full_analysis_data.csv\n")
        f.write(f"Classification Report: {timestamp}_classification_report.txt\n")
        f.write(f"Focus Areas Analysis: {timestamp}_focus_areas_analysis.csv\n")
        f.write(f"Topic Keywords: {timestamp}_topic_keywords.txt\n\n")

        f.write("VISUALIZATIONS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Sentiment Distribution: {timestamp}_sentiment_distribution.png\n")
        f.write(f"Topic Perplexity: {timestamp}_topic_perplexity.png\n")
        for i in range(optimal_topics):
            f.write(f"Topic {i + 1} WordCloud: {timestamp}_topic_{i + 1}_wordcloud.png\n")
        f.write(f"t-SNE Topics: {timestamp}_tsne_topics.png\n")
        f.write(f"t-SNE Sentiments: {timestamp}_tsne_sentiments.png\n")
        f.write(f"Topic Sentiment Distribution: {timestamp}_topic_sentiment_distribution.png\n")
        f.write(f"Confusion Matrix: {timestamp}_confusion_matrix.png\n")
        f.write(f"Feature Importance: {timestamp}_feature_importance.png\n")
        f.write(f"Focus Areas Analysis: {timestamp}_focus_areas_analysis.png\n")

    # Close the Snowflake connection
    conn.close()
    print(f"Analysis complete! Summary report saved to {summary_path}")

except Exception as e:
    print(f"An error occurred during analysis: {str(e)}")

    # Attempt to close connection if it exists
    try:
        if 'conn' in locals():
            conn.close()
            print("Snowflake connection closed")
    except:
        pass