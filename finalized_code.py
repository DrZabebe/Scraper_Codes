import snowflake.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os
import re
from wordcloud import WordCloud




# Snowflake connection configuration
def connect_to_snowflake():
    conn = snowflake.connector.connect(
        user='',
        password='',
        account='',
        warehouse='',
        database='',
        schema=''
    )
    return conn


# Function to fetch data from Snowflake
def fetch_data_from_snowflake(conn, table_name):
    cursor = conn.cursor()

    try:
        query = f"""
        SELECT AUTHOR, TITLE, GAPS, CATEGORY, YEAR, LINK 
        FROM {table_name}
        """
        cursor.execute(query)

        # Fetch column names
        column_names = [desc[0] for desc in cursor.description]

        # Fetch all rows
        data = cursor.fetchall()

        # Create pandas DataFrame
        df = pd.DataFrame(data, columns=column_names)

        cursor.close()
        print(f"Fetched {len(df)} rows from {table_name}.")
        return df
    except Exception as e:
        print(f"Error fetching data from {table_name}: {e}")
        cursor.close()
        return pd.DataFrame()


# Main function to retrieve data
def retrieve_data():
    try:
        # Connect to Snowflake
        conn = connect_to_snowflake()
        print("Connected to Snowflake successfully.")

        # Fetch data from both tables
        elsevier_df = fetch_data_from_snowflake(conn, "ELSEVIER_ARTICLES_STAGING")
        pubmed_df = fetch_data_from_snowflake(conn, "PUBMED_ARTICLES_STAGING")

        # Close connection
        conn.close()

        return elsevier_df, pubmed_df
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(), pd.DataFrame()


# Call the function
elsevier_df, pubmed_df = retrieve_data()

# Basic info about the retrieved data
print("\nElsevier DataFrame Info:")
print(f"Number of rows: {len(elsevier_df)}")
print(f"Columns: {elsevier_df.columns.tolist()}")

print("\nPubMed DataFrame Info:")
print(f"Number of rows: {len(pubmed_df)}")
print(f"Columns: {pubmed_df.columns.tolist()}")


def analyze_basic_stats(df, source_name):
    """Calculate basic statistics for a dataset"""
    print(f"\nAnalyzing basic statistics for {source_name}...")

    if df.empty:
        print(f"No data available for {source_name}")
        return None

    # Record count
    record_count = len(df)

    # Missing value counts
    missing_values = {col: int(df[col].isna().sum()) for col in df.columns}
    completeness = {col: (1 - missing_values[col] / record_count) * 100 for col in df.columns}

    # Year range (if year column exists and has data)
    year_range = None
    if 'YEAR' in df.columns and not df['YEAR'].isna().all():
        min_year = df['YEAR'].min()
        max_year = df['YEAR'].max()
        year_range = (min_year, max_year)

    # Return statistics dictionary
    return {
        'record_count': record_count,
        'missing_values': missing_values,
        'completeness': completeness,
        'year_range': year_range
    }


# Calculate basic statistics
elsevier_stats = analyze_basic_stats(elsevier_df, "Elsevier")
pubmed_stats = analyze_basic_stats(pubmed_df, "PubMed")

# Print basic statistics
if elsevier_stats:
    print("\nElsevier Basic Statistics:")
    print(f"Total records: {elsevier_stats['record_count']}")
    print("Data completeness:")
    for col, pct in elsevier_stats['completeness'].items():
        print(f"  {col}: {pct:.1f}%")
    if elsevier_stats['year_range']:
        print(f"Year range: {elsevier_stats['year_range'][0]} - {elsevier_stats['year_range'][1]}")

if pubmed_stats:
    print("\nPubMed Basic Statistics:")
    print(f"Total records: {pubmed_stats['record_count']}")
    print("Data completeness:")
    for col, pct in pubmed_stats['completeness'].items():
        print(f"  {col}: {pct:.1f}%")
    if pubmed_stats['year_range']:
        print(f"Year range: {pubmed_stats['year_range'][0]} - {pubmed_stats['year_range'][1]}")



# Function to analyze categories (research areas)
def analyze_categories(df, source_name, output_dir="./peer_review_analysis"):
    """Analyze the distribution of categories/research areas"""
    print(f"\nAnalyzing categories for {source_name}...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have category data
    if 'CATEGORY' not in df.columns or df.empty:
        print(f"No category data available for {source_name}")
        return None

    # Process categories (they might be in multiple formats)
    all_categories = []
    for categories in df['CATEGORY'].dropna():
        if isinstance(categories, str):
            # Split by common delimiters
            if ',' in categories:
                all_categories.extend([c.strip() for c in categories.split(',')])
            elif ';' in categories:
                all_categories.extend([c.strip() for c in categories.split(';')])
            elif '|' in categories:
                all_categories.extend([c.strip() for c in categories.split('|')])
            else:
                all_categories.append(categories.strip())

    # Count occurrences of each category
    category_counts = Counter(all_categories)

    # Get top categories and their counts
    top_categories = category_counts.most_common(20)

    # Calculate category statistics
    total_categories = len(category_counts)
    top_5_count = sum(count for _, count in top_categories[:5])
    total_count = sum(category_counts.values())
    top_5_percent = (top_5_count / total_count * 100) if total_count > 0 else 0

    # Create visualization - Horizontal bar chart of top 15 categories
    plt.figure(figsize=(12, 8))

    categories = [category for category, _ in top_categories][:15]
    counts = [count for _, count in top_categories][:15]

    # Reverse for horizontal bar chart
    categories.reverse()
    counts.reverse()

    # Color palette
    colors = sns.color_palette("viridis", len(categories))

    # Create the horizontal bar chart
    bars = plt.barh(categories, counts, color=colors)

    # Add data labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{int(width)}', ha='left', va='center')

    plt.title(f'Top 15 Research Categories ({source_name})', fontsize=16)
    plt.xlabel('Number of Publications', fontsize=12)
    plt.tight_layout()

    # Save the chart with high resolution
    chart_path = os.path.join(output_dir, f'{source_name.lower()}_top_categories.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()

    # Create Pie Chart of top 5 vs others
    plt.figure(figsize=(10, 10))

    # Get top 5 categories and sum of all others
    top_5_labels = [cat for cat, _ in top_categories[:5]]
    top_5_values = [count for _, count in top_categories[:5]]
    other_value = total_count - sum(top_5_values)

    # Add "Other" to labels and values
    labels = top_5_labels + ['Other Categories']
    values = top_5_values + [other_value]

    # Color palette for pie chart
    pie_colors = sns.color_palette("viridis", len(labels))

    # Create the pie chart
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=pie_colors)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(f'Top 5 Categories vs Others ({source_name})', fontsize=16)

    # Save the pie chart
    pie_chart_path = os.path.join(output_dir, f'{source_name.lower()}_category_distribution.png')
    plt.savefig(pie_chart_path, dpi=300)
    plt.close()

    # Return analysis results and paths to the charts
    return {
        'top_categories': top_categories,
        'total_unique_categories': total_categories,
        'top_5_percent': top_5_percent,
        'bar_chart_path': chart_path,
        'pie_chart_path': pie_chart_path
    }


# Function to compare categories between databases
def compare_categories(elsevier_analysis, pubmed_analysis, output_dir="./peer_review_analysis"):
    """Compare categories between Elsevier and PubMed databases"""

    if not elsevier_analysis or not pubmed_analysis:
        print("Cannot compare categories - missing analysis data")
        return None

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get top 10 categories from each database
    elsevier_top10 = dict(elsevier_analysis['top_categories'][:10])
    pubmed_top10 = dict(pubmed_analysis['top_categories'][:10])

    # Find common categories
    common_categories = set(elsevier_top10.keys()).intersection(set(pubmed_top10.keys()))

    # Create a side-by-side comparison chart
    plt.figure(figsize=(15, 10))

    # Identify all unique categories in the top 10 of either database
    all_categories = sorted(set(elsevier_top10.keys()).union(set(pubmed_top10.keys())))

    # Get counts for each database (0 if category not in top 10)
    elsevier_counts = [elsevier_top10.get(cat, 0) for cat in all_categories]
    pubmed_counts = [pubmed_top10.get(cat, 0) for cat in all_categories]

    # Set up the bar chart
    x = np.arange(len(all_categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 10))
    rects1 = ax.bar(x - width / 2, elsevier_counts, width, label='Elsevier', color='blue', alpha=0.7)
    rects2 = ax.bar(x + width / 2, pubmed_counts, width, label='PubMed', color='red', alpha=0.7)

    # Add some text for labels, title and custom x-axis tick labels
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Category Comparison: Elsevier vs PubMed', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(all_categories, rotation=45, ha='right')
    ax.legend()

    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:  # Only add labels to non-zero bars
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    # Save the comparison chart
    comparison_path = os.path.join(output_dir, 'category_comparison.png')
    plt.savefig(comparison_path, dpi=300)
    plt.close()

    # Return comparison results
    return {
        'common_categories': list(common_categories),
        'comparison_chart_path': comparison_path
    }


# Test the functions with our data
elsevier_category_analysis = analyze_categories(elsevier_df, "Elsevier")
pubmed_category_analysis = analyze_categories(pubmed_df, "PubMed")

# Print analysis results
if elsevier_category_analysis:
    print("\nElsevier Category Analysis:")
    print(f"Total unique categories: {elsevier_category_analysis['total_unique_categories']}")
    print(f"Top 5 categories represent {elsevier_category_analysis['top_5_percent']:.1f}% of all mentions")
    print("\nTop 10 categories:")
    for category, count in elsevier_category_analysis['top_categories'][:10]:
        print(f"  {category}: {count}")
    print(f"\nBar chart saved to: {elsevier_category_analysis['bar_chart_path']}")
    print(f"Pie chart saved to: {elsevier_category_analysis['pie_chart_path']}")

if pubmed_category_analysis:
    print("\nPubMed Category Analysis:")
    print(f"Total unique categories: {pubmed_category_analysis['total_unique_categories']}")
    print(f"Top 5 categories represent {pubmed_category_analysis['top_5_percent']:.1f}% of all mentions")
    print("\nTop 10 categories:")
    for category, count in pubmed_category_analysis['top_categories'][:10]:
        print(f"  {category}: {count}")
    print(f"\nBar chart saved to: {pubmed_category_analysis['bar_chart_path']}")
    print(f"Pie chart saved to: {pubmed_category_analysis['pie_chart_path']}")

# Compare categories if both analyses exist
if elsevier_category_analysis and pubmed_category_analysis:
    category_comparison = compare_categories(elsevier_category_analysis, pubmed_category_analysis)

    if category_comparison:
        print("\nCategory Comparison:")
        print(f"Common categories in top 10: {', '.join(category_comparison['common_categories'])}")
        print(f"Comparison chart saved to: {category_comparison['comparison_chart_path']}")



# Function to analyze publication years
# Function to analyze publication years
def analyze_publication_years(df, source_name, output_dir="./peer_review_analysis"):
    """Analyze the distribution of publication years and trends"""
    print(f"\nAnalyzing publication years for {source_name}...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have year data
    if 'YEAR' not in df.columns or df.empty:
        print(f"No year data available for {source_name}")
        return None

    # Convert YEAR to numeric (handling any non-numeric values)
    df['YEAR_NUMERIC'] = pd.to_numeric(df['YEAR'], errors='coerce')

    # Count publications by year
    year_counts = df['YEAR_NUMERIC'].value_counts().sort_index()

    # Calculate year-over-year growth
    yoy_growth = {}
    for i in range(1, len(year_counts.index)):
        prev_year = year_counts.index[i - 1]
        curr_year = year_counts.index[i]
        prev_count = year_counts[prev_year]
        curr_count = year_counts[curr_year]
        growth_pct = ((curr_count - prev_count) / prev_count * 100) if prev_count > 0 else float('inf')
        yoy_growth[curr_year] = growth_pct

    # Get year range
    min_year = year_counts.index.min() if not year_counts.empty else None
    max_year = year_counts.index.max() if not year_counts.empty else None

    # Calculate publication rate
    total_years = max_year - min_year + 1 if min_year and max_year else 0
    avg_pubs_per_year = df.shape[0] / total_years if total_years > 0 else 0


    # Calculate moving average (3-year window)
    years = list(year_counts.index)
    counts = list(year_counts.values)
    moving_avg = []

    for i in range(len(years)):
        # For each year, calculate average of that year and two previous years
        if i >= 2:
            avg = sum(counts[i - 2:i + 1]) / 3
        elif i == 1:
            avg = sum(counts[i - 1:i + 1]) / 2
        else:
            avg = counts[i]
        moving_avg.append(avg)

    # Create publications by year chart
    plt.figure(figsize=(12, 6))
    plt.bar(years, counts, color='skyblue', alpha=0.7, label='Annual Publications')
    plt.plot(years, moving_avg, color='navy', linewidth=2, marker='o', label='3-Year Moving Average')

    plt.title(f'Publications by Year ({source_name})', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Publications', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the chart
    year_chart_path = os.path.join(output_dir, f'{source_name.lower()}_publications_by_year.png')
    plt.savefig(year_chart_path, dpi=300)
    plt.close()

    # Create year-over-year growth chart
    if yoy_growth:
        plt.figure(figsize=(12, 6))

        growth_years = list(yoy_growth.keys())
        growth_values = list(yoy_growth.values())

        # Filter out infinite growth (first appearance of a category)
        valid_indices = [i for i, value in enumerate(growth_values) if not np.isinf(value)]
        valid_years = [growth_years[i] for i in valid_indices]
        valid_values = [growth_values[i] for i in valid_indices]

        # Create colored bars (green for positive, red for negative growth)
        colors = ['green' if x >= 0 else 'red' for x in valid_values]
        plt.bar(valid_years, valid_values, color=colors, alpha=0.7)

        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f'Year-over-Year Growth ({source_name})', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Growth (%)', fontsize=12)
        plt.xticks(rotation=45)

        # Add value labels on bars
        for i, v in enumerate(valid_values):
            plt.text(valid_years[i], v + (5 if v >= 0 else -10),
                     f'{v:.1f}%', ha='center', fontsize=9)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the chart
        growth_chart_path = os.path.join(output_dir, f'{source_name.lower()}_growth_by_year.png')
        plt.savefig(growth_chart_path, dpi=300)
        plt.close()
    else:
        growth_chart_path = None

    # Create heatmap of categories over time (if 'CATEGORY' column exists)
    category_year_heatmap_path = None
    if 'CATEGORY' in df.columns:
        # Extract categories
        all_categories = []
        category_year_counts = {}

        # Group by year
        for year in sorted(df['YEAR_NUMERIC'].dropna().unique()):
            year_df = df[df['YEAR_NUMERIC'] == year]

            # Extract categories for this year
            year_categories = []
            for categories in year_df['CATEGORY'].dropna():
                if isinstance(categories, str):
                    if ',' in categories:
                        year_categories.extend([c.strip() for c in categories.split(',')])
                    elif ';' in categories:
                        year_categories.extend([c.strip() for c in categories.split(';')])
                    elif '|' in categories:
                        year_categories.extend([c.strip() for c in categories.split('|')])
                    else:
                        year_categories.append(categories.strip())

            # Count categories for this year
            year_counts = Counter(year_categories)

            # Store category counts for this year
            category_year_counts[year] = year_counts

            # Add to overall category list
            all_categories.extend(year_categories)

        # Get top 10 categories overall
        top_categories = [cat for cat, _ in Counter(all_categories).most_common(10)]

        if top_categories and len(category_year_counts) > 1:
            # Create data for heatmap
            years_for_heatmap = sorted(category_year_counts.keys())
            heatmap_data = []

            for cat in top_categories:
                row = []
                for year in years_for_heatmap:
                    row.append(category_year_counts[year].get(cat, 0))
                heatmap_data.append(row)

            # Create heatmap
            plt.figure(figsize=(12, 8))
            ax = sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu",
                             xticklabels=years_for_heatmap, yticklabels=top_categories)
            plt.title(f'Category Trends Over Time ({source_name})', fontsize=16)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Category', fontsize=12)

            # Save the heatmap
            category_year_heatmap_path = os.path.join(output_dir, f'{source_name.lower()}_category_year_heatmap.png')
            plt.savefig(category_year_heatmap_path, dpi=300)
            plt.close()

    return {
        'year_counts': dict(year_counts),  # Changed from year_counts.to_dict()
        'yoy_growth': yoy_growth,
        'min_year': min_year,
        'max_year': max_year,
        'avg_pubs_per_year': avg_pubs_per_year,
        'year_chart_path': year_chart_path,
        'growth_chart_path': growth_chart_path,
        'category_year_heatmap_path': category_year_heatmap_path
    }

# Function to compare publication timelines between databases
def compare_publication_timelines(elsevier_analysis, pubmed_analysis, output_dir="./peer_review_analysis"):
    """Compare publication timelines between Elsevier and PubMed databases"""

    if not elsevier_analysis or not pubmed_analysis:
        print("Cannot compare timelines - missing analysis data")
        return None

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get year counts from both databases
    elsevier_years = elsevier_analysis['year_counts']
    pubmed_years = pubmed_analysis['year_counts']

    # Combine all years
    all_years = sorted(set(list(elsevier_years.keys()) + list(pubmed_years.keys())))

    # Get counts for each database (0 if year not present)
    elsevier_counts = [elsevier_years.get(year, 0) for year in all_years]
    pubmed_counts = [pubmed_years.get(year, 0) for year in all_years]

    # Create a combined timeline chart
    plt.figure(figsize=(14, 7))

    width = 0.35  # Width of bars
    x = np.arange(len(all_years))

    plt.bar(x - width / 2, elsevier_counts, width, label='Elsevier', color='blue', alpha=0.7)
    plt.bar(x + width / 2, pubmed_counts, width, label='PubMed', color='red', alpha=0.7)

    plt.title('Publication Timeline Comparison', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Publications', fontsize=12)
    plt.xticks(x, all_years, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the comparison chart
    comparison_path = os.path.join(output_dir, 'publication_timeline_comparison.png')
    plt.savefig(comparison_path, dpi=300)
    plt.close()

    # Calculate combined growth trend
    combined_counts = {}
    for year in all_years:
        combined_counts[year] = elsevier_years.get(year, 0) + pubmed_years.get(year, 0)

    # Calculate overall growth trend (linear regression)
    years_numeric = np.array([int(year) for year in all_years])
    counts_numeric = np.array(list(combined_counts.values()))

    # Only calculate trend if we have enough years
    trend_chart_path = None
    if len(years_numeric) > 2:
        # Create trend analysis chart
        plt.figure(figsize=(12, 6))

        # Plot actual data points
        plt.scatter(years_numeric, counts_numeric, color='green', alpha=0.7, label='Combined Publications')

        # Calculate and plot trendline
        z = np.polyfit(years_numeric, counts_numeric, 1)
        p = np.poly1d(z)
        plt.plot(years_numeric, p(years_numeric), "r--", label=f'Trend (slope: {z[0]:.2f})')

        plt.title('Overall Publication Growth Trend', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Number of Publications', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the trend chart
        trend_chart_path = os.path.join(output_dir, 'publication_trend_analysis.png')
        plt.savefig(trend_chart_path, dpi=300)
        plt.close()

    # Return comparison results
    return {
        'combined_timeline': combined_counts,
        'comparison_chart_path': comparison_path,
        'trend_chart_path': trend_chart_path
    }


# Test the functions with our data
elsevier_timeline_analysis = analyze_publication_years(elsevier_df, "Elsevier")
pubmed_timeline_analysis = analyze_publication_years(pubmed_df, "PubMed")

# Print analysis results
if elsevier_timeline_analysis:
    print("\nElsevier Timeline Analysis:")
    print(
        f"Publication year range: {elsevier_timeline_analysis['min_year']} - {elsevier_timeline_analysis['max_year']}")
    print(f"Average publications per year: {elsevier_timeline_analysis['avg_pubs_per_year']:.2f}")
    print(f"\nTimeline chart saved to: {elsevier_timeline_analysis['year_chart_path']}")
    if elsevier_timeline_analysis['growth_chart_path']:
        print(f"Growth chart saved to: {elsevier_timeline_analysis['growth_chart_path']}")
    if elsevier_timeline_analysis['category_year_heatmap_path']:
        print(f"Category timeline heatmap saved to: {elsevier_timeline_analysis['category_year_heatmap_path']}")

if pubmed_timeline_analysis:
    print("\nPubMed Timeline Analysis:")
    print(f"Publication year range: {pubmed_timeline_analysis['min_year']} - {pubmed_timeline_analysis['max_year']}")
    print(f"Average publications per year: {pubmed_timeline_analysis['avg_pubs_per_year']:.2f}")
    print(f"\nTimeline chart saved to: {pubmed_timeline_analysis['year_chart_path']}")
    if pubmed_timeline_analysis['growth_chart_path']:
        print(f"Growth chart saved to: {pubmed_timeline_analysis['growth_chart_path']}")
    if pubmed_timeline_analysis['category_year_heatmap_path']:
        print(f"Category timeline heatmap saved to: {pubmed_timeline_analysis['category_year_heatmap_path']}")

# Compare timelines if both analyses exist
if elsevier_timeline_analysis and pubmed_timeline_analysis:
    timeline_comparison = compare_publication_timelines(elsevier_timeline_analysis, pubmed_timeline_analysis)

    if timeline_comparison:
        print("\nTimeline Comparison:")
        print(f"Comparison chart saved to: {timeline_comparison['comparison_chart_path']}")
        if timeline_comparison['trend_chart_path']:
            print(f"Trend analysis saved to: {timeline_comparison['trend_chart_path']}")


# Function to analyze research gaps
def analyze_research_gaps(df, source_name, output_dir="./peer_review_analysis"):
    """Analyze the research gaps identified in the literature"""
    print(f"\nAnalyzing research gaps for {source_name}...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have gaps data
    if 'GAPS' not in df.columns or df.empty:
        print(f"No gaps data available for {source_name}")
        return None

    # Process research gaps (they might be in multiple formats)
    all_gaps = []
    for gaps in df['GAPS'].dropna():
        if isinstance(gaps, str):
            # Split by common delimiters
            if ',' in gaps:
                all_gaps.extend([g.strip() for g in gaps.split(',')])
            elif ';' in gaps:
                all_gaps.extend([g.strip() for g in gaps.split(';')])
            elif '|' in gaps:
                all_gaps.extend([g.strip() for g in gaps.split('|')])
            else:
                all_gaps.append(gaps.strip())

    # Count occurrences of each gap
    gap_counts = Counter(all_gaps)

    # Get top gaps
    top_gaps = gap_counts.most_common(20)

    # Calculate gap statistics
    total_gaps = len(gap_counts)
    papers_with_gaps = df['GAPS'].notna().sum()
    papers_with_gaps_pct = (papers_with_gaps / df.shape[0] * 100) if df.shape[0] > 0 else 0

    # Create visualization - Horizontal bar chart of top 15 gaps
    plt.figure(figsize=(12, 8))

    if top_gaps:
        gaps = [gap for gap, _ in top_gaps][:15]
        counts = [count for _, count in top_gaps][:15]

        # Reverse for horizontal bar chart
        gaps.reverse()
        counts.reverse()

        # Color palette with gradient
        colors = sns.color_palette("YlOrRd", len(gaps))

        # Create the horizontal bar chart
        bars = plt.barh(gaps, counts, color=colors)

        # Add data labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{int(width)}', ha='left', va='center')

        plt.title(f'Top 15 Research Gaps ({source_name})', fontsize=16)
        plt.xlabel('Frequency', fontsize=12)
        plt.tight_layout()

        # Save the chart
        gap_chart_path = os.path.join(output_dir, f'{source_name.lower()}_top_gaps.png')
        plt.savefig(gap_chart_path, dpi=300)
        plt.close()
    else:
        gap_chart_path = None

    # Create a word cloud of all gaps
    wordcloud_path = None
    if all_gaps:
        # Concatenate all gaps into one string
        text = ' '.join(all_gaps)

        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              colormap='plasma', max_words=100, contour_width=1)
        wordcloud.generate(text)

        # Display the word cloud
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Research Gaps Word Cloud ({source_name})', fontsize=16)
        plt.tight_layout()

        # Save the word cloud
        wordcloud_path = os.path.join(output_dir, f'{source_name.lower()}_gaps_wordcloud.png')
        plt.savefig(wordcloud_path, dpi=300)
        plt.close()

    # Create a chart showing gap mentions over time (if YEAR column is available)
    gap_timeline_path = None
    if 'YEAR' in df.columns and len(top_gaps) > 0:
        # Get top 5 gaps
        top5_gaps = [gap for gap, _ in top_gaps[:5]]

        # Count mentions of each top gap by year
        years = sorted(df['YEAR'].dropna().unique())
        gap_by_year = {gap: [] for gap in top5_gaps}

        for year in years:
            year_df = df[df['YEAR'] == year]

            for gap in top5_gaps:
                # Count mentions of this gap in this year
                count = 0
                for gaps_text in year_df['GAPS'].dropna():
                    if isinstance(gaps_text, str):
                        if gap in gaps_text:
                            count += 1

                gap_by_year[gap].append(count)

        # Create timeline chart
        plt.figure(figsize=(12, 6))

        for gap in top5_gaps:
            plt.plot(years, gap_by_year[gap], marker='o', linewidth=2, label=gap)

        plt.title(f'Research Gap Mentions Over Time ({source_name})', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Number of Mentions', fontsize=12)
        plt.xticks(years, rotation=45)
        plt.legend(title='Research Gaps')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the chart
        gap_timeline_path = os.path.join(output_dir, f'{source_name.lower()}_gap_timeline.png')
        plt.savefig(gap_timeline_path, dpi=300)
        plt.close()

    # Analyze connection between gaps and categories (if CATEGORY column exists)
    gap_category_heatmap_path = None
    if 'CATEGORY' in df.columns and len(top_gaps) > 0:
        # Get top 5 gaps and top 5 categories
        top5_gaps = [gap for gap, _ in top_gaps[:5]]

        # Process categories
        all_categories = []
        for categories in df['CATEGORY'].dropna():
            if isinstance(categories, str):
                if ',' in categories:
                    all_categories.extend([c.strip() for c in categories.split(',')])
                elif ';' in categories:
                    all_categories.extend([c.strip() for c in categories.split(';')])
                elif '|' in categories:
                    all_categories.extend([c.strip() for c in categories.split('|')])
                else:
                    all_categories.append(categories.strip())

        # Get top 5 categories
        top5_categories = [cat for cat, _ in Counter(all_categories).most_common(5)]

        # Create co-occurrence matrix
        cooccurrence = np.zeros((len(top5_gaps), len(top5_categories)))

        for i, row in df.iterrows():
            gaps_text = row['GAPS']
            categories_text = row['CATEGORY']

            if pd.notna(gaps_text) and pd.notna(categories_text) and isinstance(gaps_text, str) and isinstance(
                    categories_text, str):
                # Extract gaps from this row
                row_gaps = []
                if ',' in gaps_text:
                    row_gaps = [g.strip() for g in gaps_text.split(',')]
                elif ';' in gaps_text:
                    row_gaps = [g.strip() for g in gaps_text.split(';')]
                elif '|' in gaps_text:
                    row_gaps = [g.strip() for g in gaps_text.split('|')]
                else:
                    row_gaps = [gaps_text.strip()]

                # Extract categories from this row
                row_categories = []
                if ',' in categories_text:
                    row_categories = [c.strip() for c in categories_text.split(',')]
                elif ';' in categories_text:
                    row_categories = [c.strip() for c in categories_text.split(';')]
                elif '|' in categories_text:
                    row_categories = [c.strip() for c in categories_text.split('|')]
                else:
                    row_categories = [categories_text.strip()]

                # Check for co-occurrences
                for gap_idx, gap in enumerate(top5_gaps):
                    if any(gap in g for g in row_gaps):
                        for cat_idx, cat in enumerate(top5_categories):
                            if any(cat in c for c in row_categories):
                                cooccurrence[gap_idx, cat_idx] += 1

        # Create heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(cooccurrence, annot=True, fmt="d", cmap="YlGnBu",
                         xticklabels=top5_categories, yticklabels=top5_gaps)
        plt.title(f'Research Gaps by Category ({source_name})', fontsize=16)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Research Gap', fontsize=12)
        plt.tight_layout()

        # Save the heatmap
        gap_category_heatmap_path = os.path.join(output_dir, f'{source_name.lower()}_gap_category_heatmap.png')
        plt.savefig(gap_category_heatmap_path, dpi=300)
        plt.close()

    # Return analysis results
    return {
        'top_gaps': top_gaps,
        'total_unique_gaps': total_gaps,
        'papers_with_gaps': papers_with_gaps,
        'papers_with_gaps_pct': papers_with_gaps_pct,
        'gap_chart_path': gap_chart_path,
        'wordcloud_path': wordcloud_path,
        'gap_timeline_path': gap_timeline_path,
        'gap_category_heatmap_path': gap_category_heatmap_path
    }


# Function to compare gaps between databases
def compare_research_gaps(elsevier_analysis, pubmed_analysis, output_dir="./peer_review_analysis"):
    """Compare research gaps between Elsevier and PubMed databases"""

    if not elsevier_analysis or not pubmed_analysis:
        print("Cannot compare gaps - missing analysis data")
        return None

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get top 10 gaps from each database
    elsevier_top10 = dict(elsevier_analysis['top_gaps'][:10])
    pubmed_top10 = dict(pubmed_analysis['top_gaps'][:10])

    # Find common gaps
    common_gaps = set(elsevier_top10.keys()).intersection(set(pubmed_top10.keys()))

    # Create a comparison chart for common gaps
    common_gaps_chart_path = None
    if common_gaps:
        plt.figure(figsize=(12, len(common_gaps) * 0.8))

        common_gaps_list = list(common_gaps)
        elsevier_counts = [elsevier_top10.get(gap, 0) for gap in common_gaps_list]
        pubmed_counts = [pubmed_top10.get(gap, 0) for gap in common_gaps_list]

        # Set up the bar chart
        x = np.arange(len(common_gaps_list))
        width = 0.35

        plt.barh(x - width / 2, elsevier_counts, width, label='Elsevier', color='blue', alpha=0.7)
        plt.barh(x + width / 2, pubmed_counts, width, label='PubMed', color='red', alpha=0.7)

        plt.yticks(x, common_gaps_list)
        plt.xlabel('Frequency', fontsize=12)
        plt.title('Common Research Gaps: Elsevier vs PubMed', fontsize=16)
        plt.legend()
        plt.tight_layout()

        # Save the comparison chart
        common_gaps_chart_path = os.path.join(output_dir, 'common_gaps_comparison.png')
        plt.savefig(common_gaps_chart_path, dpi=300)
        plt.close()

    # Create a comprehensive comparison chart
    plt.figure(figsize=(14, 10))

    # Combine top gaps from both databases
    all_gaps = sorted(set(list(elsevier_top10.keys()) + list(pubmed_top10.keys())))

    # Get counts for each database (0 if gap not in top 10)
    elsevier_counts = [elsevier_top10.get(gap, 0) for gap in all_gaps]
    pubmed_counts = [pubmed_top10.get(gap, 0) for gap in all_gaps]

    # Set up the bar chart
    x = np.arange(len(all_gaps))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, len(all_gaps) * 0.5))
    rects1 = ax.barh(x - width / 2, elsevier_counts, width, label='Elsevier', color='blue', alpha=0.7)
    rects2 = ax.barh(x + width / 2, pubmed_counts, width, label='PubMed', color='red', alpha=0.7)

    # Add labels and title
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title('All Top Research Gaps Comparison', fontsize=16)
    ax.set_yticks(x)
    ax.set_yticklabels(all_gaps)
    ax.legend()

    # Add value labels
    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            if width > 0:  # Only add labels to non-zero bars
                ax.annotate(f'{int(width)}',
                            xy=(width + 0.5, rect.get_y() + rect.get_height() / 2),
                            xytext=(0, 0),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='left', va='center')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    # Save the comprehensive comparison chart
    comprehensive_chart_path = os.path.join(output_dir, 'research_gaps_comparison.png')
    plt.savefig(comprehensive_chart_path, dpi=300)
    plt.close()

    # Return comparison results
    return {
        'common_gaps': list(common_gaps),
        'common_gaps_chart_path': common_gaps_chart_path,
        'comprehensive_chart_path': comprehensive_chart_path
    }


# Test the functions with our data
elsevier_gaps_analysis = analyze_research_gaps(elsevier_df, "Elsevier")
pubmed_gaps_analysis = analyze_research_gaps(pubmed_df, "PubMed")

# Print analysis results
if elsevier_gaps_analysis:
    print("\nElsevier Research Gaps Analysis:")
    print(f"Total unique gaps identified: {elsevier_gaps_analysis['total_unique_gaps']}")
    print(
        f"Papers with identified gaps: {elsevier_gaps_analysis['papers_with_gaps']} ({elsevier_gaps_analysis['papers_with_gaps_pct']:.1f}%)")
    print("\nTop 10 research gaps:")
    for gap, count in elsevier_gaps_analysis['top_gaps'][:10]:
        print(f"  {gap}: {count}")
    if elsevier_gaps_analysis['gap_chart_path']:
        print(f"\nGap chart saved to: {elsevier_gaps_analysis['gap_chart_path']}")
    if elsevier_gaps_analysis['wordcloud_path']:
        print(f"Word cloud saved to: {elsevier_gaps_analysis['wordcloud_path']}")
    if elsevier_gaps_analysis['gap_timeline_path']:
        print(f"Gap timeline saved to: {elsevier_gaps_analysis['gap_timeline_path']}")
    if elsevier_gaps_analysis['gap_category_heatmap_path']:
        print(f"Gap-category heatmap saved to: {elsevier_gaps_analysis['gap_category_heatmap_path']}")

if pubmed_gaps_analysis:
    print("\nPubMed Research Gaps Analysis:")
    print(f"Total unique gaps identified: {pubmed_gaps_analysis['total_unique_gaps']}")
    print(
        f"Papers with identified gaps: {pubmed_gaps_analysis['papers_with_gaps']} ({pubmed_gaps_analysis['papers_with_gaps_pct']:.1f}%)")
    print("\nTop 10 research gaps:")
    for gap, count in pubmed_gaps_analysis['top_gaps'][:10]:
        print(f"  {gap}: {count}")
    if pubmed_gaps_analysis['gap_chart_path']:
        print(f"\nGap chart saved to: {pubmed_gaps_analysis['gap_chart_path']}")
    if pubmed_gaps_analysis['wordcloud_path']:
        print(f"Word cloud saved to: {pubmed_gaps_analysis['wordcloud_path']}")
    if pubmed_gaps_analysis['gap_timeline_path']:
        print(f"Gap timeline saved to: {pubmed_gaps_analysis['gap_timeline_path']}")
    if pubmed_gaps_analysis['gap_category_heatmap_path']:
        print(f"Gap-category heatmap saved to: {pubmed_gaps_analysis['gap_category_heatmap_path']}")

# Compare gaps if both analyses exist
if elsevier_gaps_analysis and pubmed_gaps_analysis:
    gaps_comparison = compare_research_gaps(elsevier_gaps_analysis, pubmed_gaps_analysis)

    if gaps_comparison:
        print("\nResearch Gaps Comparison:")
        print(f"Common gaps in top lists: {len(gaps_comparison['common_gaps'])}")
        print("Common gaps include:")
        for gap in gaps_comparison['common_gaps']:
            print(f"  {gap}")
        print(f"\nCommon gaps chart saved to: {gaps_comparison['common_gaps_chart_path']}")
        print(f"Comprehensive comparison chart saved to: {gaps_comparison['comprehensive_chart_path']}")

