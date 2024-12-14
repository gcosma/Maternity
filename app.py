import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from utils.topic_model import TopicModeler

class JudiciaryReportScraper:
    def __init__(self, base_url='https://www.judiciary.uk/'):
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_search_url(self, search_term='maternity', 
                       report_type='', 
                       order='relevance', 
                       start_date=None, 
                       end_date=None):
        """Construct the search URL with provided parameters"""
        params = {
            's': search_term,
            'pfd_report_type': report_type,
            'post_type': 'pfd',
            'order': order
        }

        # Handle date filtering if provided
        if start_date:
            params.update({
                'after-day': start_date.day,
                'after-month': start_date.month,
                'after-year': start_date.year
            })

        if end_date:
            params.update({
                'before-day': end_date.day,
                'before-month': end_date.month,
                'before-year': end_date.year
            })

        # Convert params to URL query string
        query_string = '&'.join(f'{k}={v}' for k, v in params.items())
        return f'{self.base_url}?{query_string}'

    def scrape_report_links(self, url):
        """Scrape report links from the search results page"""
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Target the specific selector for report links
            report_links = soup.select('h3.entry-title a')
            
            # Extract link details
            report_details = []
            for link in report_links:
                report_details.append({
                    'title': link.text.strip(),
                    'url': link['href']
                })
            
            return report_details
        
        except Exception as e:
            st.error(f"Error scraping report links: {e}")
            return []

    def extract_report_text(self, url):
        """Extract full text from a report page"""
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract main content - adjust selector as needed
            content = soup.select_one('div.entry-content')
            
            if content:
                return content.get_text(strip=True)
            return ""
        
        except Exception as e:
            st.error(f"Error extracting text from {url}: {e}")
            return ""

def main():
    st.title("Judiciary UK Report Analyzer")
    
    # Sidebar for configuration
    st.sidebar.header("Scraping Configuration")
    search_term = st.sidebar.text_input("Search Term", "maternity")
    
    # Date range selection
    start_date = st.sidebar.date_input("Start Date", None)
    end_date = st.sidebar.date_input("End Date", None)
    
    # Order selection
    order = st.sidebar.selectbox(
        "Sort Order", 
        ["relevance", "date"]
    )
    
    # Number of topics
    num_topics = st.sidebar.slider("Number of Topics", 2, 10, 5)
    
    # Scrape button
    if st.sidebar.button("Analyze Reports"):
        with st.spinner("Scraping and Analyzing Reports..."):
            # Initialize scraper
            scraper = JudiciaryReportScraper()
            
            # Construct search URL
            search_url = scraper.get_search_url(
                search_term=search_term, 
                order=order,
                start_date=start_date,
                end_date=end_date
            )
            
            # Scrape report links
            report_links = scraper.scrape_report_links(search_url)
            
            if not report_links:
                st.error("No reports found!")
                return
            
            # Extract texts
            texts = [scraper.extract_report_text(link['url']) for link in report_links]
            
            # Perform topic modeling
            modeler = TopicModeler()
            
            # Scikit-learn LDA
            st.subheader("Scikit-learn LDA Topic Model")
            lda_output, topics = modeler.lda_sklearn(texts, num_topics)
            
            for idx, topic in enumerate(topics):
                st.write(f"Topic {idx + 1}: {', '.join(topic)}")
            
            # Display report links
            st.subheader(f"Found {len(report_links)} Reports")
            for link in report_links:
                st.write(f"[{link['title']}]({link['url']})")

if __name__ == "__main__":
    main()
