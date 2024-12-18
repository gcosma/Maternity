import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import requests
from bs4 import BeautifulSoup
import time
import urllib3
import io
import pdfplumber
import logging
import os
import zipfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import ast

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Referer': 'https://judiciary.uk/'
}

def make_request(url: str, retries: int = 3, delay: int = 2) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            time.sleep(delay)
            response = requests.get(url, headers=HEADERS, verify=False, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"Request failed: {str(e)}")
                raise e
            time.sleep(delay * (attempt + 1))
    return None

def clean_text(text: str) -> str:
    if not text:
        return ""
    
    try:
        text = str(text)
        text = unicodedata.normalize('NFKD', text)
        
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€¦': '...',
            'â€"': '-',
            'â€¢': '•',
            'Â': ' ',
            '\u200b': '',
            '\uf0b7': '',
            '\u2019': "'",
            '\u201c': '"',
            '\u201d': '"',
            '\u2013': '-',
            '\u2022': '•'
        }
        
        for encoded, replacement in replacements.items():
            text = text.replace(encoded, replacement)
        
        text = re.sub(r'<[^>]+>', '', text)
        text = ''.join(char if char.isprintable() or char == '\n' else ' ' for char in text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    except Exception as e:
        logging.error(f"Error in clean_text: {e}")
        return ""

def extract_metadata(content: str) -> dict:
    metadata = {
        'date_of_report': None,
        'ref': None,
        'deceased_name': None,
        'coroner_name': None,
        'coroner_area': None,
        'categories': []
    }
    
    if not content:
        return metadata
        
    try:
        date_patterns = [
            r'Date of report:\s*(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})',
            r'Date of report:\s*(\d{1,2}/\d{1,2}/\d{4})',
            r'DATED this (\d{1,2}(?:st|nd|rd|th)?\s+day of [A-Za-z]+\s+\d{4})',
            r'Date:\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})'
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, content, re.IGNORECASE)
            if date_match:
                date_str = date_match.group(1)
                try:
                    if '/' in date_str:
                        date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                    else:
                        date_str = re.sub(r'(?<=\d)(st|nd|rd|th)', '', date_str)
                        date_str = re.sub(r'day of ', '', date_str)
                        try:
                            date_obj = datetime.strptime(date_str, '%d %B %Y')
                        except ValueError:
                            date_obj = datetime.strptime(date_str, '%d %b %Y')
                    
                    metadata['date_of_report'] = date_obj.strftime('%d/%m/%Y')
                    break
                except ValueError as e:
                    logging.warning(f"Invalid date format found: {date_str} - {e}")
        
        ref_match = re.search(r'Ref(?:erence)?:?\s*([-\d]+)', content)
        if ref_match:
            metadata['ref'] = ref_match.group(1).strip()
        
        name_match = re.search(r'Deceased name:?\s*([^\n]+)', content)
        if name_match:
            metadata['deceased_name'] = clean_text(name_match.group(1)).strip()
        
        coroner_match = re.search(r'Coroner(?:\'?s)? name:?\s*([^\n]+)', content)
        if coroner_match:
            metadata['coroner_name'] = clean_text(coroner_match.group(1)).strip()
        
        area_match = re.search(r'Coroner(?:\'?s)? Area:?\s*([^\n]+)', content)
        if area_match:
            metadata['coroner_area'] = clean_text(area_match.group(1)).strip()
        
        cat_match = re.search(r'Category:?\s*([^\n]+)', content)
        if cat_match:
            categories = cat_match.group(1).split('|')
            metadata['categories'] = [clean_text(cat).strip() for cat in categories if clean_text(cat).strip()]
        
        return metadata
        
    except Exception as e:
        logging.error(f"Error extracting metadata: {e}")
        return metadata

def get_pfd_categories() -> List[str]:
    return [
        "accident-at-work-and-health-and-safety-related-deaths",
        "alcohol-drug-and-medication-related-deaths",
        "care-home-health-related-deaths",
        "child-death-from-2015",
        "community-health-care-and-emergency-services-related-deaths",
        "emergency-services-related-deaths-2019-onwards",
        "hospital-death-clinical-procedures-and-medical-management-related-deaths",
        "mental-health-related-deaths",
        "other-related-deaths",
        "police-related-deaths",
        "product-related-deaths",
        "railway-related-deaths",
        "road-highways-safety-related-deaths",
        "service-personnel-related-deaths",
        "state-custody-related-deaths",
        "suicide-from-2015",
        "wales-prevention-of-future-deaths-reports-2019-onwards"
    ]

def save_pdf(pdf_url: str, base_dir: str = 'pdfs') -> Tuple[Optional[str], Optional[str]]:
    try:
        os.makedirs(base_dir, exist_ok=True)
        
        response = make_request(pdf_url)
        if not response:
            return None, None
        
        filename = os.path.basename(pdf_url)
        filename = re.sub(r'[^\w\-_\. ]', '_', filename)
        local_path = os.path.join(base_dir, filename)
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        return local_path, filename
    
    except Exception as e:
        logging.error(f"Error saving PDF {pdf_url}: {e}")
        return None, None

def extract_pdf_content(pdf_path: str, chunk_size: int = 10) -> str:
    try:
        filename = os.path.basename(pdf_path)
        text_chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for i in range(0, len(pdf.pages), chunk_size):
                chunk = pdf.pages[i:i+chunk_size]
                chunk_text = "\n\n".join([page.extract_text() or "" for page in chunk])
                text_chunks.append(chunk_text)
                
        full_content = f"PDF FILENAME: {filename}\n\n{''.join(text_chunks)}"
        return clean_text(full_content)
        
    except Exception as e:
        logging.error(f"Error extracting PDF text from {pdf_path}: {e}")
        return ""

def get_report_content(url: str) -> Optional[Dict]:
    try:
        logging.info(f"Fetching content from: {url}")
        response = make_request(url)
        if not response:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find('div', class_='flow') or soup.find('article', class_='single__post')
        
        if not content:
            logging.warning(f"No content found at {url}")
            return None
        
        paragraphs = content.find_all(['p', 'table'])
        webpage_text = '\n\n'.join(p.get_text(strip=True, separator=' ') for p in paragraphs)
        
        pdf_contents = []
        pdf_paths = []
        pdf_names = []
        
        pdf_links = (
            soup.find_all('a', class_='related-content__link', href=re.compile(r'\.pdf$')) or
            soup.find_all('a', href=re.compile(r'\.pdf$'))
        )
        
        for pdf_link in pdf_links:
            pdf_url = pdf_link['href']
            
            if not pdf_url.startswith(('http://', 'https://')):
                pdf_url = f"https://www.judiciary.uk{pdf_url}" if not pdf_url.startswith('/') else f"https://www.judiciary.uk/{pdf_url}"
            
            pdf_path, pdf_name = save_pdf(pdf_url)
            
            if pdf_path:
                pdf_content = extract_pdf_content(pdf_path)
                pdf_contents.append(pdf_content)
                pdf_paths.append(pdf_path)
                pdf_names.append(pdf_name)
        
        return {
            'content': clean_text(webpage_text),
            'pdf_contents': pdf_contents,
            'pdf_paths': pdf_paths,
            'pdf_names': pdf_names
        }
        
    except Exception as e:
        logging.error(f"Error getting report content: {e}")
        return None

def scrape_page(url: str) -> List[Dict]:
    try:
        response = make_request(url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results_list = soup.find('ul', class_='search__list')
        
        if not results_list:
            logging.warning(f"No results list found on page: {url}")
            return []
        
        reports = []
        cards = results_list.find_all('div', class_='card')
        
        for card in cards:
            try:
                title_elem = card.find('h3', class_='card__title').find('a')
                if not title_elem:
                    continue
                
                title = clean_text(title_elem.text)
                card_url = title_elem['href']
                
                logging.info(f"Processing report: {title}")
                
                if not card_url.startswith(('http://', 'https://')):
                    card_url = f"https://www.judiciary.uk{card_url}"
                
                content_data = get_report_content(card_url)
                
                if content_data:
                    report = {
                        'Title': title,
                        'URL': card_url,
                        'Content': content_data['content']
                    }
                    
                    for i, (name, content, path) in enumerate(zip(
                        content_data['pdf_names'],
                        content_data['pdf_contents'],
                        content_data['pdf_paths']
                    ), 1):
                        report[f'PDF_{i}_Name'] = name
                        report[f'PDF_{i}_Content'] = content
                        report[f'PDF_{i}_Path'] = path
                    
                    reports.append(report)
                    logging.info(f"Successfully processed: {title}")
                
            except Exception as e:
                logging.error(f"Error processing card: {e}")
                continue
        
        return reports
        
    except Exception as e:
        logging.error(f"Error fetching page {url}: {e}")
        return []

def get_total_pages(url: str) -> int:
    try:
        response = make_request(url)
        if not response:
            return 0
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        pagination = soup.find('nav', class_='navigation pagination')
        if pagination:
            page_numbers = pagination.find_all('a', class_='page-numbers')
            numbers = [int(p.text.strip()) for p in page_numbers if p.text.strip().isdigit()]
            if numbers:
                return max(numbers)
        
        results = soup.find('ul', class_='search__list')
        if results and results.find_all('div', class_='card'):
            return 1
            
        return 0
        
    except Exception as e:
        logging.error(f"Error getting total pages: {e}")
        return 0

def scrape_pfd_reports(keyword: Optional[str] = None,
                      category: Optional[str] = None,
                      date_after: Optional[str] = None,
                      date_before: Optional[str] = None,
                      order: str = "relevance",
                      max_pages: Optional[int]
def scrape_pfd_reports(keyword: Optional[str] = None,
                      category: Optional[str] = None,
                      date_after: Optional[str] = None,
                      date_before: Optional[str] = None,
                      order: str = "relevance",
                      max_pages: Optional[int] = None) -> List[Dict]:
    all_reports = []
    current_page = 1
    base_url = "https://www.judiciary.uk"
    
    params = {
        'post_type': 'pfd',
        'order': order
    }
    
    if keyword and keyword.strip():
        params['s'] = keyword.strip()
    if category:
        params['pfd_report_type'] = category
    
    if date_after:
        try:
            day, month, year = date_after.split('/')
            params['after-year'] = year
            params['after-month'] = month
            params['after-day'] = day
        except ValueError as e:
            logging.error(f"Invalid date_after format: {e}")
            return []
    
    if date_before:
        try:
            day, month, year = date_before.split('/')
            params['before-year'] = year
            params['before-month'] = month
            params['before-day'] = day
        except ValueError as e:
            logging.error(f"Invalid date_before format: {e}")
            return []
    
    param_strings = [f"{k}={v}" for k, v in params.items()]
    initial_url = f"{base_url}/?{'&'.join(param_strings)}"
    
    st.write(f"Searching URL: {initial_url}")
    
    try:
        total_pages = get_total_pages(initial_url)
        if total_pages == 0:
            st.warning("No results found")
            return []
            
        logging.info(f"Total pages to scrape: {total_pages}")
        
        if max_pages:
            total_pages = min(total_pages, max_pages)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while current_page <= total_pages:
            page_url = initial_url if current_page == 1 else f"{base_url}/page/{current_page}/?{'&'.join(param_strings)}"
            
            status_text.text(f"Scraping page {current_page} of {total_pages}...")
            progress_bar.progress(current_page / total_pages)
            
            reports = scrape_page(page_url)
            
            if reports:
                all_reports.extend(reports)
                logging.info(f"Found {len(reports)} reports on page {current_page}")
            else:
                logging.warning(f"No reports found on page {current_page}")
                if current_page > 1:
                    break
            
            current_page += 1
        
        progress_bar.progress(1.0)
        status_text.text(f"Completed! Total reports found: {len(all_reports)}")
        
        return all_reports
    
    except Exception as e:
        logging.error(f"Error in scrape_pfd_reports: {e}")
        st.error(f"An error occurred while scraping reports: {e}")
        return []

def process_scraped_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        metadata = df['Content'].fillna("").apply(extract_metadata)
        metadata_df = pd.DataFrame(metadata.tolist())
        result = pd.concat([df, metadata_df], axis=1)
        
        try:
            result['date_of_report'] = pd.to_datetime(
                result['date_of_report'],
                format='%d/%m/%Y',
                errors='coerce'
            )
        except Exception as e:
            logging.error(f"Error converting dates: {e}")
        
        return result
            
    except Exception as e:
        logging.error(f"Error in process_scraped_data: {e}")
        return df

def plot_timeline(df: pd.DataFrame) -> None:
    timeline_data = df.groupby(
        pd.Grouper(key='date_of_report', freq='ME')
    ).size().reset_index()
    timeline_data.columns = ['Date', 'Count']
    
    fig = px.line(timeline_data, x='Date', y='Count',
                  title='Reports Timeline',
                  labels={'Count': 'Number of Reports'})
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Reports",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_category_distribution(df: pd.DataFrame) -> None:
    all_cats = []
    for cats in df['categories'].dropna():
        if isinstance(cats, list):
            all_cats.extend(cats)
    
    cat_counts = pd.Series(all_cats).value_counts()
    
    fig = px.bar(
        x=cat_counts.index,
        y=cat_counts.values,
        title='Category Distribution',
        labels={'x': 'Category', 'y': 'Count'}
    )
    
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Number of Reports",
        xaxis={'tickangle': 45}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_coroner_areas(df: pd.DataFrame) -> None:
    area_counts = df['coroner_area'].value_counts().head(20)
    
    fig = px.bar(
        x=area_counts.index,
        y=area_counts.values,
        title='Top 20 Coroner Areas',
        labels={'x': 'Area', 'y': 'Count'}
    )
    
    fig.update_layout(
        xaxis_title="Coroner Area",
        yaxis_title="Number of Reports",
        xaxis={'tickangle': 45}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def extract_topics_lda(df: pd.DataFrame, num_topics: int = 5, max_features: int = 1000) -> Tuple[LatentDirichletAllocation, TfidfVectorizer, np.ndarray]:
    try:
        texts = []
        for _, row in df.iterrows():
            content_parts = []
            
            if pd.notna(row.get('Content')):
                content_parts.append(str(row['Content']))
            
            if pd.notna(row.get('Title')):
                content_parts.append(str(row['Title']))
            
            pdf_columns = [col for col in df.columns if col.endswith('_Content')]
            for pdf_col in pdf_columns:
                if pd.notna(row.get(pdf_col)):
                    content_parts.append(str(row[pdf_col]))
            
            if content_parts:
                cleaned_text = ' '.join(clean_text_for_modeling(text) for text in content_parts)
                if cleaned_text.strip():
                    texts.append(cleaned_text)
        
        if not texts:
            raise ValueError("No valid text content found after preprocessing")
            
        logging.info(f"Processing {len(texts)} documents for topic modeling")
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.95,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        logging.info("Creating document-term matrix...")
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        if tfidf_matrix.shape[1] == 0:
            raise ValueError("No valid terms found after vectorization")
        
        logging.info(f"Document-term matrix shape: {tfidf_matrix.shape}")
        
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=20,
            learning_method='batch',
            n_jobs=-1,
            doc_topic_prior=0.1,
            topic_word_prior=0.01
        )
        
        logging.info("Fitting LDA model...")
        doc_topic_dist = lda_model.fit_transform(tfidf_matrix)
        
        logging.info("Topic modeling completed successfully")
        return lda_model, vectorizer, doc_topic_dist
    
    except Exception as e:
        logging.error(f"Error in topic extraction: {e}")
        raise e

def clean_text_for_modeling(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        text = text.lower()
        text = ''.join(char for char in text if ord(char) < 128)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        
        custom_stop_words = {
            'report', 'pfd', 'death', 'deaths', 'deceased', 'coroner', 'coroners',
            'date', 'ref', 'name', 'area', 'regulation', 'paragraph', 'section',
            'prevention', 'future', 'investigation', 'inquest', 'circumstances',
            'response', 'duty', 'action', 'actions', 'concern', 'concerns', 'trust',
            'hospital', 'service', 'services', 'chief', 'executive', 'family',
            'dear', 'sincerely', 'following', 'report', 'reports', 'days'
        }
        stop_words.update(custom_stop_words)
        
        filtered_tokens = []
        for token in tokens:
            if (len(token) > 2 and
                not token.isnumeric() and
                not all(c.isdigit() or c == '/' for c in token) and
                token not in stop_words):
                filtered_tokens.append(token)
        
        if not filtered_tokens:
            return ""
            
        return ' '.join(filtered_tokens)
    
    except Exception as e:
        logging.error(f"Error cleaning text for modeling: {e}")
        return ""

def create_network_diagram(topic_words: List[str], 
                         tfidf_matrix: np.ndarray, 
                         similarity_threshold: float = 0.3) -> go.Figure:
    try:
        similarities = cosine_similarity(tfidf_matrix)
        G = nx.Graph()
        
        for i, word1 in enumerate(topic_words):
            for j, word2 in enumerate(topic_words[i+1:], i+1):
                similarity = similarities[i][j]
                if similarity >= similarity_threshold:
                    G.add_edge(word1, word2, weight=similarity)
        
        pos = nx.spring_layout(G)
        
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            hoverinfo='text',
            text=[],
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=20
            ))
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
            
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            node_trace['text'] += (node,)
        
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=0,l=0,r=0,t=0)
                     ))
        
        return fig
    
    except Exception as e:
        logging.error(f"Error creating network diagram: {e}")
        return None

def render_scraping_tab():
    st.header("Scrape PFD Reports")
    
    if 'scraped_data' in st.session_state and st.session_state.scraped_data is not None:
        st.success(f"Found {len(st.session_state.scraped_data)} reports")
        
        st.subheader("Results")
        st.dataframe(
            st.session_state.scraped_data,
            column_config={
                "URL": st.column_config.LinkColumn("Report Link"),
                "date_of_report": st.column_config.DateColumn("Date of Report"),
                "categories": st.column_config.ListColumn("Categories")
            },
            hide_index=True
        )
        
        show_export_options(st.session_state.scraped_data, "scraped")
    
    with st.form("scraping_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            search_keyword = st.text_input("Search keywords:", "")
            category = st.selectbox("PFD Report type:", [""] + get_pfd_categories())
            order = st.selectbox("Sort by:", [
                "relevance",
                "desc",
                "asc"
            ], format_func=lambda x: {
                "relevance": "Relevance",
                "desc": "Newest first",
                "asc": "Oldest first"
            }[x])
        
        with col2:
            date_after = st.date_input(
                "Published after:",
                None,
                format="DD/MM/YYYY"
            )
            
            date_before = st.date_input(
                "Published before:",
                None,
                format="DD/MM/YYYY"
            )
            
            max_pages = st.number_input(
                "Maximum pages to scrape (0 for all):", 
                min_value=0, 
                value=0
            )
        
        submitted = st.form_submit_button("Search Reports")
    
    if submitted:
        try:
            with st.spinner("Searching for reports..."):
                date_after_str = date_after.strftime('%d/%m/%Y') if date_after else None
                date_before_str = date_before.strftime('%d/%m/%Y') if date_before else None
                max_pages_val = None if max_pages == 0 else max_pages
                
                reports = scrape_pfd_reports(
                    keyword=search_keyword,
                    category=category if category else None,
                    date_after=date_after_str,
                    date_before=date_before_str,
                    order=order,
                    max_pages=max_pages_val
                )
                
                if reports:
                    df = pd.DataFrame(reports)
                    df = process_scraped_data(df)
                    
                    if not df.empty:
                        st.session_state.current_data = None
                        st.session_state.scraped_data = None
                        st.session_state.uploaded_data = None
                        st.session_state.data_source = None

                        st.session_state.scraped_data = df.copy()
                        st.session_state.data_source = 'scraped'
                        st.session_state.current_data = df.copy()
                        
                        st.rerun()
                    else:
                        st.warning("Scraping completed, but no valid data was found.")
                else:
                    st.warning("No reports found matching your search criteria")
                    return False
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"Scraping error: {e}")
            return False

def show_export_options(df: pd.DataFrame, prefix: str):
    st.subheader("Export Options")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pfd_reports_{prefix}_{timestamp}"
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Reports (CSV)",
            csv,
            f"{filename}.csv",
            "text/csv",
            key=f"download_csv_{prefix}_{timestamp}"
        )
    
    with col2:
        excel_data = export_to_excel(df)
        st.download_button(
            "📥 Download Reports (Excel)",
            excel_data,
            f"{filename}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_excel_{prefix}_{timestamp}"
        )
    
    st.subheader("Download PDFs")
    if st.button(f"Download all PDFs_{timestamp}", key=f"pdf_button_{prefix}_{timestamp}"):
        with st.spinner("Preparing PDF download..."):
            pdf_zip_path = f"{filename}_pdfs.zip"
            
            with zipfile.ZipFile(pdf_zip_path, 'w') as zipf:
                unique_pdfs = set()
                pdf_columns = [col for col in df.columns if col.startswith('PDF_') and col.endswith('_Path')]
                
                for col in pdf_columns:
                    paths = df[col].dropna()
                    unique_pdfs.update(paths)
                
                for pdf_path in unique_pdfs:
                    if pdf_path and os.path.exists(pdf_path):
                        zipf.write(pdf_path, os.path.basename(pdf_path))
            
            with open(pdf_zip_path, 'rb') as f:
                st.download_button(
                    "📦 Download All PDFs (ZIP)",
                    f.read(),
                    pdf_zip_path,
                    "application/zip",
                    key=f"download_pdfs_zip_{prefix}_{timestamp}"
                )
            
            os.remove(pdf_zip_path)

def validate_data(data: pd.DataFrame, purpose: str = "analysis") -> Tuple[bool, str]:
    if data is None:
        return False, "No data available. Please scrape or upload data first."
    
    if not isinstance(data, pd.DataFrame):
        return False, "Invalid data format. Expected pandas DataFrame."
    
    if len(data) == 0:
        return False, "Dataset is empty."
        
    if purpose == "analysis":
        required_columns = ['date_of_report', 'categories', 'coroner_area']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
            
    elif purpose == "topic_modeling":
        if 'Content' not in data.columns:
            return False, "Missing required column: Content"
            
        valid_docs = data['Content'].dropna().str.strip().str.len() > 0
        if valid_docs.sum() < 2:
            return False, "Not enough valid documents found. Please ensure you have documents with text content."
            
    if 'date_of_report' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['date_of_report']):
        try:
            pd.to_datetime(data['date_of_report'])
        except Exception:
            return False, "Invalid date format in date_of_report column."
            
    if 'categories' in data.columns:
        if not data['categories'].apply(lambda x: isinstance(x, (list, type(None)))).all():
            return False, "Categories must be stored as lists or None values."
    
    return True, "Data is valid"

def analyze_data_quality(df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        completeness = 100 - (df.isnull().sum().sum() / (len(df.columns) * len(df)) * 100)
        st.metric("Data Completeness", f"{completeness:.2f}%")
    
    with col3:
        duplicates = df.duplicated().sum()
        st.metric("Duplicate Records", duplicates)
    
    with col4:
        unique_records = len(df) - duplicates
        st.metric("Unique Records", unique_records)
    
    st.subheader("Column Completeness")
    
    missing_data = df.isnull().sum()
    missing_percentages = (missing_data / len(df)) * 100
    
    completeness_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Values': missing_data.values,
        'Completeness (%)': (100 - missing_percentages).round(2)
    }).sort_values('Missing Values', ascending=False)
    
    fig_completeness = px.bar(
        completeness_df,
        x='Column',
        y='Completeness (%)',
        title='Column Completeness',
        labels={'Completeness (%)': 'Completeness (%)'},
        color='Completeness (%)',
        color_continuous_scale='RdYlGn'
    )
    fig_completeness.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_completeness, use_container_width=True)
    
    st.subheader("Detailed Column Analysis")
    
    tab1, tab2, tab3 = st.tabs([
        "Categorical Columns", 
        "Numerical Columns", 
        "Date Columns"
    ])
    
    with tab1:
        categorical_cols = ['categories', 'coroner_area']
        
        for col in categorical_cols:
            if col in df.columns:
                if col == 'categories':
                    all_categories = []
                    for cats in df[col].dropna():
                        if isinstance(cats, str):
                            try:
                                parsed_cats = ast.literal_eval(cats)
                                if isinstance(parsed_cats, list):
                                    all_categories.extend(parsed_cats)
                                else:
                                    all_categories.append(parsed_cats)
                            except:
                                all_categories.append(cats)
                        elif isinstance(cats, list):
                            all_categories.extend(cats)
                    
                    category_counts = pd.Series(all_categories).value_counts()
                else:
                    category_counts = df[col].value_counts()
                
                top_categories = category_counts.head(10)
                
                fig_cat = px.bar(
                    x=top_categories.index, 
                    y=top_categories.values,
                    title=f'Top 10 Categories in {col}',
                    labels={'x': 'Category', 'y': 'Count'}
                )
                fig_cat.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_cat, use_container_width=True)
    
    with tab2:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_cols) > 0:
            desc_stats = df[numerical_cols].describe()
            st.dataframe(desc_stats)
            
            fig_box = go.Figure()
            for col in numerical_cols:
                fig_box.add_trace(go.Box(y=df[col], name=col))
            
            fig_box.update_layout(
                title='Distribution of Numerical Columns',
                yaxis_title='Values'
            )
            st.plotly_chart(fig_box, use_container_width=True)
    
    with tab3:
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        for col in date_cols:
            min_date = df[col].min()
            max_date = df[col].max()
            
            monthly_dist = df.groupby(pd.Grouper(key=col, freq='ME')).size()
            
            fig_date = px.line(
                x=monthly_dist.index, 
                y=monthly_dist.values,
                title=f'Monthly Distribution of {col}',
                labels={'x': 'Date', 'y': 'Number of Records'}
            )
            st.plotly_chart(fig_date, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Earliest Date", min_date.strftime('%Y-%m-%d'))
            with col2:
                st.metric("Latest Date", max_date.strftime('%Y-%m-%d'))
            with col3:
                st.metric("Total Date Range", f"{(max_date - min_date).days} days")

def render_analysis_tab(data: pd.DataFrame):
    st.header("Reports Analysis")
    
    if st.session_state.current_data is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            total_reports = len(st.session_state.current_data) if isinstance(st.session_state.current_data, pd.DataFrame) else 0
            data_source = st.session_state.data_source or "unknown source"
            st.info(f"Currently analyzing {total_reports} reports from {data_source}")
        with col2:
            if st.button("Clear Current Data"):
                st.session_state.current_data = None
                st.session_state.data_source = None
                st.session_state.scraped_data = None
                st.session_state.uploaded_data = None
                st.rerun()
    
    if st.session_state.current_data is None:
        upload_col1, upload_col2 = st.columns([3, 1])
        with upload_col1:
            uploaded_file = st.file_uploader(
                "Upload CSV or Excel file", 
                type=['csv', 'xlsx']
            )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.lower().endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file type")
                    return
                
                required_columns = [
                    'Title', 'URL', 'Content', 
                    'date_of_report', 'categories', 'coroner_area'
                ]
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    st.write("Available columns:", list(df.columns))
                    return
                
                processed_df = process_scraped_data(df)
                
                st.session_state.uploaded_data = processed_df.copy()
                st.session_state.current_data = processed_df.copy()
                st.session_state.data_source = 'uploaded'
                
                st.success(f"File uploaded successfully! Total reports: {len(processed_df)}")
                st.rerun()
                
            except Exception as read_error:
                st.error(f"Error reading file: {read_error}")
                logging.error(f"File read error: {read_error}", exc_info=True)
                return
        return
    
    try:
        is_valid, message = validate_data(data, "analysis")
        if not is_valid:
            st.error(message)
            return
            
        min_date = data['date_of_report'].min().date()
        max_date = data['date_of_report'].max().date()
        
        with st.sidebar:
            st.header("Analysis Filters")
            
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="date_range_filter"
            )
            
            all_categories = set()
            for cats in data['categories'].dropna():
                if isinstance(cats, list):
                    all_categories.update(cats)
            
            selected_categories = st.multiselect(
                "Categories",
                options=sorted(all_categories),
                key="categories_filter"
            )
            
            coroner_areas = sorted(data['coroner_area'].dropna().unique())
            selected_areas = st.multiselect(
                "Coroner Areas",
                options=coroner_areas,
                key="areas_filter"
            )
        
        filtered_df = data.copy()
        
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['date_of_report'].dt.date >= date_range[0]) &
                (filtered_df['date_of_report'].dt.date <= date_range[1])
            ]
        
        if selected_categories:
            filtered_df = filtered_df[
                filtered_df['categories'].apply(
                    lambda x: bool(x) and any(cat in x for cat in selected_categories)
                )
            ]
        
        if selected_areas:
            filtered_df = filtered_df[filtered_df['coroner_area'].isin(selected_areas)]
        
        active_filters = []
        if len(date_range) == 2 and (date_range[0] != min_date or date_range[1] != max_date):
            active_filters.append(f"Date range: {date_range[0]} to {date_range[1]}")
        if selected_categories:
            active_filters.append(f"Categories: {', '.join(selected_categories)}")
        if selected_areas:
            active_filters.append(f"Areas: {', '.join(selected_areas)}")
            
        if active_filters:
            st.info(f"Active filters: {' • '.join(active_filters)}")
        
        if len(filtered_df) == 0:
            st.warning("No data matches the selected filters.")
            return
            
        st.write(f"Showing {len(filtered_df)} of {len(data)} reports")
        
        st.subheader("Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reports", len(filtered_df))
        with col2:
            st.metric("Unique Coroner Areas", filtered_df['coroner_area'].nunique())
        with col3:
            categories_count = sum(len(cats) if isinstance(cats, list) else 0 
                                 for cats in filtered_df['categories'].dropna())
            st.metric("Total Category Tags", categories_count)
        with col4:
            date_range_days = (filtered_df['date_of_report'].max() - filtered_df['date_of_report'].min()).days
            avg_reports_month = len(filtered_df) / (date_range_days / 30) if date_range_days > 0 else len(filtered_df)
            st.metric("Avg Reports/Month", f"{avg_reports_month:.1f}")
        
        st.subheader("Visualizations")
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Timeline",
            "Categories",
            "Coroner Areas",
            "Data Quality"
        ])
        
        with viz_tab1:
            try:
                plot_timeline(filtered_df)
            except Exception as e:
                st.error(f"Error creating timeline plot: {str(e)}")
        
        with viz_tab2:
            try:
                plot_category_distribution(filtered_df)
            except Exception as e:
                st.error(f"Error creating category distribution plot: {str(e)}")
        
        with viz_tab3:
            try:
                plot_coroner_areas(filtered_df)
            except Exception as e:
                st.error(f"Error creating coroner areas plot: {str(e)}")
                
        with viz_tab4:
            try:
                analyze_data_quality(filtered_df)
            except Exception as e:
                st.error(f"Error creating data quality analysis: {str(e)}")
        
        show_export_options(filtered_df, "filtered")
        
    except Exception as e:
        st.error(f"An unexpected error occurred in the analysis tab: {str(e)}")
        logging.error(f"Unexpected error in render_analysis_tab: {e}", exc_info=True)

def export_to_excel(df: pd.DataFrame) -> bytes:
    excel_buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        return excel_buffer.getvalue()
    finally:
        excel_buffer.close()

def initialize_session_state():
    if not hasattr(st.session_state, 'initialized') or not st.session_state.initialized:
        default_state = {
            'data_source': None,
            'current_data': None,
            'scraped_data': None,
            'uploaded_data': None,
            'topic_model': None,
            'cleanup_done': False,
            'last_scrape_time': None,
            'last_upload_time': None,
            'analysis_filters': {
                'date_range': None,
                'selected_categories': [],
                'selected_areas': []
            },
            'topic_model_settings': {
                'num_topics': 5, 'max_features': 1000,
                'similarity_threshold': 0.3
            },
            'initialized': True
        }
        
        for key, value in default_state.items():
            setattr(st.session_state, key, value)
    
    if not st.session_state.cleanup_done:
        try:
            pdf_dir = 'pdfs'
            os.makedirs(pdf_dir, exist_ok=True)
            
            current_time = time.time()
            cleanup_count = 0
            
            for file in os.listdir(pdf_dir):
                file_path = os.path.join(pdf_dir, file)
                try:
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > 86400:
                            os.remove(file_path)
                            cleanup_count += 1
                except Exception as e:
                    logging.warning(f"Error cleaning up file {file_path}: {e}")
            
            if cleanup_count > 0:
                logging.info(f"Cleaned up {cleanup_count} old PDF files")
            
            st.session_state.cleanup_done = True
        
        except Exception as e:
            logging.error(f"Error during PDF cleanup: {e}")
            st.session_state.cleanup_done = False

def main():
    initialize_session_state()
    st.title("UK Judiciary PFD Reports Analysis")
    st.markdown("""
    This application allows you to analyze Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    You can either scrape new reports or analyze existing data.
    """)
    
    current_tab = st.radio(
        "Select section:",
        ["🔍 Scrape Reports", "📊 Analysis", "🔬 Topic Modeling"],
        label_visibility="collapsed",
        horizontal=True,
        key="main_tab_selector"
    )
    
    st.markdown("---")
    
    if current_tab == "🔍 Scrape Reports":
        render_scraping_tab()
    
    elif current_tab == "📊 Analysis":
        if hasattr(st.session_state, 'current_data'):
            render_analysis_tab(st.session_state.current_data)
        else:
            st.warning("No data available. Please scrape reports or upload a file.")
    
    elif current_tab == "🔬 Topic Modeling":
        if not hasattr(st.session_state, 'current_data') or st.session_state.current_data is None:
            st.warning("No data available. Please scrape reports or upload a file first.")
            return
            
        try:
            is_valid, message = validate_data(st.session_state.current_data, "topic_modeling")
            if is_valid:
                render_topic_modeling_tab(st.session_state.current_data)
            else:
                st.error(message)
        except Exception as e:
            st.error(f"Error in topic modeling: {e}")
            logging.error(f"Topic modeling error: {e}", exc_info=True)
    
    st.markdown("---")
    st.markdown(
        """<div style='text-align: center'>
        <p>Built with Streamlit • Data from UK Judiciary</p>
        </div>""",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()