import streamlit as st
import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def clean_text(text):
    """
    Comprehensive text cleaning function to handle encoded characters
    """
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Specific replacements for encoded characters
    replacements = {
        'â€™': "'",   # Smart single quote right
        'â€˜': "'",   # Smart single quote left
        'â€œ': '"',   # Left double quote
        'â€': '"',    # Right double quote
        'â€¦': '...',  # Ellipsis
        'â€"': '—',   # Em dash
        'â€¢': '•',   # Bullet point
        'Â': '',      # Unwanted character
        '\u200b': '', # Zero-width space
        '\uf0b7': ''  # Private use area character
    }
    
    # Apply replacements
    for encoded, replacement in replacements.items():
        text = text.replace(encoded, replacement)
    
    # Standardize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Fix concatenated fields
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between lower and uppercase letters
    
    return text

def extract_date(text):
    """
    Comprehensive date extraction method with multiple strategies
    """
    if not text:
        return None
    
    # Clean the text first
    cleaned_text = clean_text(text)
    
    # Comprehensive date patterns
    date_patterns = [
        r'Date\s*of\s*report\s*[:]*\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})',  # Supports / . or -
        r'Date\s*of\s*report\s*:\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})',
        r'Date\s*of\s*report\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})'
    ]
    
    # Additional text-based patterns (for edge cases)
    text_patterns = [
        r'\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})\b',  # Anywhere in text
        r'\b(\d{4}[/.-]\d{1,2}[/.-]\d{1,2})\b'   # Alternate format
    ]
    
    # Combine patterns
    all_patterns = date_patterns + text_patterns
    
    # Possible date separators
    separators = ['/', '-', '.']
    
    for pattern in all_patterns:
        match = re.search(pattern, cleaned_text, re.IGNORECASE)
        if match:
            date_str = match.group(1).strip()
            
            # Try different date formats
            formats_to_try = [
                '%d/%m/%Y',   # UK/EU format: Day/Month/Year
                '%m/%d/%Y',   # US format: Month/Day/Year
                '%Y/%m/%d',   # ISO format: Year/Month/Day
                '%d-%m-%Y',   # Alternative UK format
                '%m-%d-%Y',   # Alternative US format
                '%Y-%m-%d'    # Alternative ISO format
            ]
            
            for sep in separators:
                for fmt in formats_to_try:
                    reformatted_fmt = fmt.replace('/', sep).replace('-', sep)
                    try:
                        parsed_date = datetime.strptime(date_str.replace('/', sep).replace('-', sep), reformatted_fmt)
                        # Additional validation
                        if 2000 <= parsed_date.year <= datetime.now().year:
                            return parsed_date.strftime('%d/%m/%Y')
                    except ValueError:
                        continue
    
    logging.warning(f"Could not extract valid date from text: {text}")
    return None

class MetadataExtractor:
    """Metadata extraction class specifically designed for PFD reports format"""
    
    METADATA_PATTERNS = {
        'date_of_report': [
            r'Date\s*of\s*report\s*[:]*\s*(\d{2}/\d{2}/\d{4})',
            r'Date\s*of\s*report\s*:\s*(\d{2}/\d{2}/\d{4})',
            r'Date\s*of\s*report\s*(\d{2}/\d{2}/\d{4})'
        ],
        'reference': [
            r'Ref:\s*(20\d{2}-\d{4})',
            r'Reference:\s*(20\d{2}-\d{4})'
        ],
        'deceased_name': [
            r'Deceased name:\s*([^:\n]+?)(?=\s*(?:Coroner|$))',
            r'Name of (?:the )?deceased:\s*([^:\n]+?)(?=\s*(?:Coroner|$))',
            r'^([^:]+?)(?=Coroners?\s+name:)'
        ],
        'coroner_name': [
            r'Coroners?\s*name:\s*([^:\n]+?)(?=\s*(?:Coroner Area:|$))',
            r'I am ([^,]+),\s*(?:Assistant )?Coroner',
            r'Coroners?\s*name:\s*([^:\n]+?)(?=\s*(?:Coroners?\s*Area:|$))'
        ],
        'coroner_area': [
            r'Coroners?\s*Area:\s*([^:\n]+?)(?=\s*(?:Category:|$))',
            r'for the (?:coroner )?area of\s+([^\.]+)',
            r'Coroners?\s*Area:\s*([^:\n]+?)(?=\s*(?:Category:|$))'
        ],
        'categories': [
            r'Category:\s*([^:\n]+?)(?=\s*(?:This report is being sent to:|$))',
            r'Category:\s*([^\n]+)'
        ],
        'sent_to': [
            r'This report is being sent to:\s*([^:\n]+?)(?=\s*(?:REGULATION|\d|$))',
            r'This report is being sent to:\s*([^\n]+)'
        ]
    }

    def extract_metadata(self, content: str) -> Dict:
        """Extract metadata following the exact PFD report format"""
        metadata = {
            'date_of_report': None,
            'reference': None,
            'deceased_name': None,
            'coroner_name': None,
            'coroner_area': None,
            'categories': None,
            'sent_to': None
        }
        
        if not content:
            return metadata
        
        # Clean the content thoroughly before extraction
        cleaned_content = clean_text(content)
        metadata['date_of_report'] = extract_date(content)

        # Try each pattern for each field
        for field, patterns in self.METADATA_PATTERNS.items():
            if not isinstance(patterns, list):
                patterns = [patterns]
            
            for pattern in patterns:
                match = re.search(pattern, cleaned_content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    
                    if field == 'date_of_report':
                        try:
                            # Validate date format
                            datetime.strptime(value, '%d/%m/%Y')
                            metadata[field] = value
                        except ValueError:
                            logging.warning(f"Invalid date format: {value}")
                            continue
                    
                    elif field == 'categories':
                        # Split categories on pipe and clean
                        categories = [cat.strip() for cat in value.split('|')]
                        # Remove empty categories and clean up
                        categories = [re.sub(r'\s+', ' ', cat).strip() for cat in categories if cat.strip()]
                        if categories:
                            metadata[field] = categories
                    
                    else:
                        # Clean up other values
                        value = re.sub(r'\s+', ' ', value).strip()
                        
                        # Handle special case for deceased name
                        if field == 'deceased_name' and 'Coroner' in value:
                            value = value.split('Coroner')[0].strip()
                        
                        metadata[field] = value
                    
                    break  # Stop trying patterns once we find a match
        
        return metadata

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe to extract metadata from content"""
    extractor = MetadataExtractor()
    metadata_rows = []
    
    for idx, row in df.iterrows():
        # Initialize metadata with None values
        metadata = {
            'date_of_report': None,
            'reference': None,
            'deceased_name': None,
            'coroner_name': None,
            'coroner_area': None,
            'categories': None,
            'sent_to': None,
            'title': row['Title'],
            'url': row['URL']
        }
        
        # Try to extract from main content first
        if pd.notna(row.get('Content')):
            content_metadata = extractor.extract_metadata(row['Content'])
            metadata.update({k: v for k, v in content_metadata.items() if v})
        
        # Try to extract from PDF contents if available
        pdf_columns = [col for col in df.columns if col.startswith('PDF_') and col.endswith('_Content')]
        for pdf_col in pdf_columns:
            if pd.notna(row.get(pdf_col)):
                pdf_metadata = extractor.extract_metadata(row[pdf_col])
                # Update only if we find new information
                metadata.update({k: v for k, v in pdf_metadata.items() if v and not metadata[k]})
        
        metadata_rows.append(metadata)
    
    # Create DataFrame from metadata
    processed_df = pd.DataFrame(metadata_rows)
    
    # Convert date string to datetime
    processed_df['date_of_report'] = pd.to_datetime(processed_df['date_of_report'], format='%d/%m/%Y', errors='coerce')
    
    return processed_df

def render_analysis_tab():
    st.header("Reports Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload previously exported reports (CSV/Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Load the data
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            # Process the data
            processed_df = process_data(raw_df)
            
            # Show data processing tabs
            data_tab1, data_tab2, data_tab3 = st.tabs(["Raw Data", "Processed Data", "Data Quality"])
            
            with data_tab1:
                st.subheader("Raw Imported Data")
                st.dataframe(
                    raw_df,
                    column_config={
                        "URL": st.column_config.LinkColumn("Report Link")
                    },
                    hide_index=True
                )
            
            with data_tab2:
                st.subheader("Processed Metadata")
                st.dataframe(
                    processed_df,
                    column_config={
                        "url": st.column_config.LinkColumn("Report Link"),
                        "date_of_report": st.column_config.DateColumn("Date of Report"),
                        "categories": st.column_config.ListColumn("Categories"),
                    },
                    hide_index=True
                )
            
            with data_tab3:
                st.subheader("Data Quality Metrics")
                col1, col2, col3 = st.columns(3)
                
                # Calculate completeness percentages
                completeness = {
                    field: (processed_df[field].notna().sum() / len(processed_df) * 100)
                    for field in ['date_of_report', 'reference', 'deceased_name', 'coroner_name', 
                                'coroner_area', 'categories', 'sent_to']
                }
                
                with col1:
                    st.metric("Date Extraction Rate", f"{completeness['date_of_report']:.1f}%")
                    st.metric("Reference Extraction Rate", f"{completeness['reference']:.1f}%")
                    st.metric("Name Extraction Rate", f"{completeness['deceased_name']:.1f}%")
                
                with col2:
                    st.metric("Coroner Name Rate", f"{completeness['coroner_name']:.1f}%")
                    st.metric("Coroner Area Rate", f"{completeness['coroner_area']:.1f}%")
                
                with col3:
                    st.metric("Category Extraction Rate", f"{completeness['categories']:.1f}%")
                    st.metric("Sent To Extraction Rate", f"{completeness['sent_to']:.1f}%")
            
            # Display filters
            st.subheader("Filter Processed Data")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Date range filter
                min_date = processed_df['date_of_report'].min()
                max_date = processed_df['date_of_report'].max()
                if pd.notna(min_date) and pd.notna(max_date):
                    date_range = st.date_input(
                        "Date range",
                        value=(min_date.date(), max_date.date()),
                        key="date_range"
                    )
            
            with col2:
                # Coroner area filter
                areas = sorted(processed_df['coroner_area'].dropna().unique())
                selected_area = st.multiselect("Coroner Area", areas)
            
            with col3:
                # Category filter
                all_categories = set()
                for cats in processed_df['categories'].dropna():
                    if isinstance(cats, list):
                        all_categories.update(cats)
                selected_categories = st.multiselect("Categories", sorted(all_categories))
            
            # Additional filters
            col1, col2 = st.columns(2)
            with col1:
                # Reference number filter
                ref_numbers = sorted(processed_df['reference'].dropna().unique())
                selected_ref = st.multiselect("Reference Numbers", ref_numbers)
            
            with col2:
                # Coroner name filter
                coroners = sorted(processed_df['coroner_name'].dropna().unique())
                selected_coroner = st.multiselect("Coroner Names", coroners)
            
            # Text search
            search_text = st.text_input("Search in deceased name or organizations:", "")
            
            # Apply filters
            filtered_df = processed_df.copy()
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (filtered_df['date_of_report'].dt.date >= start_date) &
                    (filtered_df['date_of_report'].dt.date <= end_date)
                ]
            
            if selected_area:
                filtered_df = filtered_df[filtered_df['coroner_area'].isin(selected_area)]
            
            if selected_categories:
                filtered_df = filtered_df[
                    filtered_df['categories'].apply(
                        lambda x: any(cat in x for cat in selected_categories) if isinstance(x, list) else False
                    )
                ]
            
            if selected_ref:
                filtered_df = filtered_df[filtered_df['reference'].isin(selected_ref)]
                
            if selected_coroner:
                filtered_df = filtered_df[filtered_df['coroner_name'].isin(selected_coroner)]
            
            if search_text:
                search_mask = (
                    filtered_df['deceased_name'].str.contains(search_text, case=False, na=False) |
                    filtered_df['sent_to'].str.contains(search_text, case=False, na=False)
                )
                filtered_df = filtered_df[search_mask]
            
            # Display analysis
            st.subheader("Analysis Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Reports", len(filtered_df))
            with col2:
                st.metric("Unique Coroner Areas", filtered_df['coroner_area'].nunique())
            with col3:
                current_year = datetime.now().year
                st.metric("Reports This Year", 
                         len(filtered_df[filtered_df['date_of_report'].dt.year == current_year]))
            with col4:
                if len(filtered_df) > 0:
                    date_range = (filtered_df['date_of_report'].max() - filtered_df['date_of_report'].min()).days
                    avg_reports_month = len(filtered_df) / (date_range / 30) if date_range > 0 else len(filtered_df)
                    st.metric("Avg Reports/Month", f"{avg_reports_month:.1f}")
            
            # Display filtered data
            st.dataframe(
                filtered_df.sort_values('date_of_report', ascending=False),
                column_config={
                    "url": st.column_config.LinkColumn("Report Link"),
                    "date_of_report": st.column_config.DateColumn("Date of Report"),
                    "categories": st.column_config.ListColumn("Categories"),
                },
                hide_index=True
            )
            
            # Export filtered data
            if st.button("Export Filtered Data"):
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Filtered Data",
                    csv,
                    "filtered_reports.csv",
                    "text/csv",
                    key="download_filtered"
                )
            
            # Visualization section
            st.subheader("Data Visualization")
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Timeline", "Categories", "Coroner Areas"])
            
            with viz_tab1:
                st.subheader("Reports Timeline")
                timeline_data = filtered_df.groupby(
                    pd.Grouper(key='date_of_report', freq='M')
                ).size().reset_index()
                timeline_data.columns = ['Date', 'Count']
                st.line_chart(timeline_data.set_index('Date'))
            
            with viz_tab2:
                st.subheader("Category Distribution")
                all_cats = []
                for cats in filtered_df['categories'].dropna():
                    if isinstance(cats, list):
                        all_cats.extend(cats)
                cat_counts = pd.Series(all_cats).value_counts()
                st.bar_chart(cat_counts)
            
            with viz_tab3:
                st.subheader("Reports by Coroner Area")
                area_counts = filtered_df['coroner_area'].value_counts()
                st.bar_chart(area_counts)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logging.error(f"Analysis error: {str(e)}", exc_info=True)
    else:
        st.info("Please upload a file to begin analysis")

if __name__ == "__main__":
    render_analysis_tab()
