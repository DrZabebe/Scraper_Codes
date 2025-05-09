import time
import os
import re
import pandas as pd
from datetime import datetime
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='linkedin_scraper_simple.log')
logger = logging.getLogger()

# Download NLTK resources
try:
    nltk.download('vader_lexicon', quiet=True)
    print("NLTK resources downloaded successfully")
except Exception as e:
    print(f"Failed to download NLTK resources: {e}")
    logger.error(f"Failed to download NLTK resources: {e}")

# Healthcare and AI terms for relevance check
HEALTHCARE_TERMS = [
    "healthcare", "medical", "clinical", "doctor", "physician", "hospital",
    "patient", "diagnosis", "treatment", "primary care", "health"
]

AI_TERMS = [
    "ai", "artificial intelligence", "machine learning", "ml", "deep learning",
    "algorithm", "neural network", "nlp", "computer vision"
]

# Barrier categories for classification
BARRIER_CATEGORIES = {
    "Trust and Transparency": [
        "black box", "lack of trust", "transparency", "explainable ai"
    ],
    "Clinical Authority": [
        "override clinical", "autonomy", "clinical judgment", "physician authority"
    ],
    "Training and Skills": [
        "training", "learning curve", "skills gap", "education"
    ],
    "Workflow Integration": [
        "workflow", "integration", "interface", "usability", "efficiency"
    ],
    "Infrastructure": [
        "infrastructure", "cost", "investment", "implementation"
    ],
    "Patient Relationship": [
        "patient trust", "human touch", "relationship", "empathy"
    ],
    "Data Quality": [
        "data quality", "bias", "incomplete data", "data integrity"
    ]
}


def setup_driver():
    """Set up and return the Selenium WebDriver"""
    try:
        # Configure Chrome options
        chrome_options = Options()

        # Uncomment to run in headless mode
        # chrome_options.add_argument("--headless")

        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")

        # Add user agent to make the browser look more like a real user
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")

        # Initialize the driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.maximize_window()

        # Set page load timeout
        driver.set_page_load_timeout(30)

        return driver
    except Exception as e:
        logger.error(f"Failed to set up WebDriver: {e}")
        print(f"Failed to set up WebDriver: {e}")
        raise


def login_to_linkedin(driver, username, password):
    """Login to LinkedIn with the provided credentials"""
    try:
        # Navigate to LinkedIn login page
        driver.get('https://www.linkedin.com/login')
        print("Navigating to LinkedIn login page...")
        time.sleep(3)

        # Fill username
        username_field = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "username"))
        )
        username_field.clear()
        username_field.send_keys(username)
        print("Username entered")

        # Fill password
        password_field = driver.find_element(By.ID, "password")
        password_field.clear()
        password_field.send_keys(password)
        print("Password entered")

        # Click login button
        login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
        login_button.click()
        print("Login button clicked")

        # Wait for the homepage to load - we'll look for the feed
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "global-nav"))
        )

        print("Successfully logged in to LinkedIn")
        return True
    except Exception as e:
        logger.error(f"Login failed: {e}")
        print(f"Login failed: {e}")
        return False


def search_for_posts(driver, keyword):
    """Search for posts related to the keyword"""
    try:
        # Navigate to content search
        search_url = f"https://www.linkedin.com/search/results/content/?keywords={keyword.replace(' ', '%20')}&origin=GLOBAL_SEARCH_HEADER"
        driver.get(search_url)
        print(f"Searching for: {keyword}")

        # Wait for search results to load - give it more time
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".search-results__list"))
            )
        except TimeoutException:
            print("Search results page structure may have changed. Continuing anyway...")

        # Wait longer for dynamic content
        time.sleep(5)

        return True
    except Exception as e:
        logger.error(f"Search failed for keyword {keyword}: {e}")
        print(f"Search failed for keyword {keyword}: {e}")
        return False


def extract_post_urls(driver, max_posts=20):
    """Extract URLs of posts from search results"""
    post_urls = []
    scroll_count = 0
    max_scrolls = 20 # Increased scrolls for more content

    try:
        # First try the most specific selector
        selector_options = [
            ".feed-shared-update-v2__content a.app-aware-link",  # Original selector
            ".search-results__list .entity-result a.app-aware-link",  # Alternative for search results
            ".search-results__container a.app-aware-link",  # More general selector
            "a.app-aware-link"  # Most general selector as fallback
        ]

        # Try each selector option
        current_selector = 0

        while len(post_urls) < max_posts and scroll_count < max_scrolls:
            # Print page source for debugging (uncomment if needed)
            # print(driver.page_source[:1000])  # Print first 1000 chars of page source

            # Try to find elements with current selector
            print(f"Trying selector: {selector_options[current_selector]}")
            elements = driver.find_elements(By.CSS_SELECTOR, selector_options[current_selector])

            if not elements and current_selector < len(selector_options) - 1:
                current_selector += 1
                print(f"No elements found, switching to selector: {selector_options[current_selector]}")
                continue

            # Extract links from found elements
            for element in elements:
                try:
                    post_url = element.get_attribute("href")

                    # Only add URLs that seem to be post URLs
                    if post_url and (
                            "linkedin.com/posts/" in post_url or "linkedin.com/feed/update/" in post_url) and post_url not in post_urls:
                        post_urls.append(post_url)
                        print(f"Found post URL: {post_url}")

                        if len(post_urls) >= max_posts:
                            break
                except Exception as e:
                    print(f"Error extracting URL: {e}")
                    continue

            # If we haven't found enough posts, scroll down and look for more
            if len(post_urls) < max_posts:
                driver.execute_script("window.scrollBy(0, 800);")
                time.sleep(3)  # Wait longer for content to load
                scroll_count += 1
                print(f"Scrolled {scroll_count} times, found {len(post_urls)} posts so far")

        print(f"Extracted {len(post_urls)} post URLs")
        return post_urls
    except Exception as e:
        logger.error(f"Failed to extract post URLs: {e}")
        print(f"Failed to extract post URLs: {e}")
        return post_urls


def extract_post_urls(driver, max_posts=10):
    post_urls = []
    scroll_count = 0
    max_scrolls = 10  # Increased scrolls for more content

    try:
        while len(post_urls) < max_posts and scroll_count < max_scrolls:
            print(f"Trying to extract post URLs, attempt #{scroll_count + 1}")
            elements = driver.find_elements(By.CSS_SELECTOR, "a.app-aware-link")  # Update this selector if necessary

            if elements:
                for element in elements:
                    post_url = element.get_attribute("href")
                    if post_url and (
                            "linkedin.com/posts/" in post_url or "linkedin.com/feed/update/" in post_url) and post_url not in post_urls:
                        post_urls.append(post_url)
                        print(f"Found post URL: {post_url}")

            if len(post_urls) >= max_posts:
                break

            # Scroll down and wait for new posts to load
            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(10)  # Wait for content to load
            scroll_count += 1
            print(f"Scrolled {scroll_count} times, found {len(post_urls)} posts so far")

        print(f"Extracted {len(post_urls)} post URLs")
        return post_urls
    except Exception as e:
        logger.error(f"Failed to extract post URLs: {e}")
        print(f"Failed to extract post URLs: {e}")
        return post_urls


def is_relevant_post(content):
    """Check if a post is relevant to AI in healthcare with more relaxed criteria"""
    if not content or len(content) < 20:  # Skip empty or very short posts
        return False

    # Convert to lowercase for case-insensitive matching
    text = content.lower()

    # More relaxed check for healthcare terms
    has_healthcare = any(term in text for term in HEALTHCARE_TERMS)

    # More relaxed check for AI terms
    has_ai = any(term in text for term in AI_TERMS)

    # Be more lenient - if the content is long enough and has either healthcare or AI terms
    # we'll consider it relevant since our search was already for AI healthcare
    if len(text) > 200:
        return has_healthcare or has_ai

    # For shorter posts, still require both
    return has_healthcare and has_ai


def analyze_sentiment(posts_df):
    """Analyze sentiment and identify barriers in posts"""
    if posts_df.empty:
        print("No posts to analyze")
        return posts_df

    # Initialize sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    results = []

    for _, row in posts_df.iterrows():
        try:
            # Get content
            content = row['Content'] if 'Content' in row and not pd.isna(row['Content']) else ""

            # Analyze sentiment
            sentiment_scores = sid.polarity_scores(content)
            compound_score = sentiment_scores['compound']

            # Determine sentiment label
            if compound_score >= 0.05:
                sentiment = "Positive"
            elif compound_score <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            # Identify barriers
            barriers = []
            for barrier, keywords in BARRIER_CATEGORIES.items():
                if any(keyword in content.lower() for keyword in keywords):
                    barriers.append(barrier)

            # Add new fields to the row
            row_data = row.to_dict()
            row_data.update({
                "Sentiment": sentiment,
                "Sentiment_Score": compound_score,
                "Barriers": "|".join(barriers) if barriers else "None",
                "Barrier_Count": len(barriers)
            })

            results.append(row_data)

        except Exception as e:
            logger.error(f"Error analyzing post: {e}")
            print(f"Error analyzing post: {e}")
            # Add the original row with default sentiment values
            row_data = row.to_dict()
            row_data.update({
                "Sentiment": "Neutral",
                "Sentiment_Score": 0.0,
                "Barriers": "None",
                "Barrier_Count": 0
            })
            results.append(row_data)

    return pd.DataFrame(results)


def main():
    print("Starting LinkedIn AI Healthcare Sentiment Scraper...")

    # REPLACE THESE VALUES with your actual LinkedIn credentials
    username = ""  # Replace with your LinkedIn email/username
    password = ""           # Replace with your LinkedIn password

    # Enhanced search keywords - more specific combinations
    search_keywords = [
        "artificial intelligence healthcare",
        "AI in medicine",
        "machine learning healthcare",
        "healthcare AI adoption",
        "AI clinical decision support",
        "AI medical diagnosis",
        "healthcare artificial intelligence examples"
    ]

    # Set up the driver
    try:
        driver = setup_driver()
    except Exception as e:
        print(f"Failed to set up WebDriver: {e}")
        return

    try:
        # Login to LinkedIn
        if not login_to_linkedin(driver, username, password):
            print("Login failed. Exiting.")
            driver.quit()
            return

        print("Waiting for 10 seconds to ensure login is complete...")
        time.sleep(10)  # Extended wait after login

        all_posts = []

        # Search for each keyword
        for keyword in search_keywords:
            if not search_for_posts(driver, keyword):
                print(f"Skipping keyword: {keyword}")
                continue

            # Extract post URLs
            post_urls = extract_post_urls(driver, max_posts=5)  # Limit to 5 posts per keyword for testing

            # If no posts were found, try a fallback method (e.g., hashtag search)
            if not post_urls:
                print(f"No post URLs found for keyword: {keyword}. Trying a direct search approach...")
                try:
                    hashtag_search = keyword.replace(" ", "").lower()
                    driver.get(f"https://www.linkedin.com/feed/hashtag/{hashtag_search}")
                    time.sleep(5)
                    post_urls = extract_post_urls(driver, max_posts=3)
                except Exception as e:
                    print(f"Fallback approach failed: {e}")

            # Extract data from each post
            for post_url in post_urls:
                post_data = extract_post_data(driver, post_url)

                if post_data:
                    # Check relevance with more lenient criteria
                    if is_relevant_post(post_data["Content"]):
                        post_data["Keyword"] = keyword
                        all_posts.append(post_data)
                        print(f"Added relevant post from {post_data['Author']}")
                    else:
                        print(f"Post not relevant. Content: {post_data['Content'][:100]}...")

                time.sleep(2)  # Small delay between posts

            time.sleep(3)  # Longer delay between keywords

        # Create DataFrame from collected posts
        posts_df = pd.DataFrame(all_posts)

        if not posts_df.empty:
            # Display basic stats
            print(f"\nCollected {len(posts_df)} relevant posts")

            # Analyze sentiment and barriers
            print("Analyzing sentiment and barriers...")
            analyzed_df = analyze_sentiment(posts_df)

            sentiment_counts = analyzed_df['Sentiment'].value_counts()
            print("\nSentiment breakdown:")
            for sentiment, count in sentiment_counts.items():
                print(f"  - {sentiment}: {count} posts")

            if 'Barriers' in analyzed_df.columns:
                all_barriers = []
                for barriers in analyzed_df['Barriers']:
                    if barriers != "None":
                        all_barriers.extend(barriers.split("|"))

                barrier_counts = {}
                for barrier in all_barriers:
                    barrier_counts[barrier] = barrier_counts.get(barrier, 0) + 1

                if barrier_counts:
                    print("\nTop barriers mentioned:")
                    sorted_barriers = sorted(barrier_counts.items(), key=lambda x: x[1], reverse=True)
                    for barrier, count in sorted_barriers[:3]:
                        print(f"  - {barrier}: {count} mentions")

            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"linkedin_healthcare_ai_{timestamp}.csv"
            analyzed_df.to_csv(output_file, index=False)
            print(f"\nData saved to {output_file}")
        else:
            print("No relevant posts found. Try different keywords or search criteria.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

    finally:
        # Close the browser
        driver.quit()
        print("Browser closed. Script completed.")

if __name__ == "__main__":
    main()
