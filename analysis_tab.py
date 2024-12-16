import streamlit as st
import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class MetadataExtractor:
    """Metadata extraction class specifically designed for PFD reports format"""
    
    METADATA_PATTERNS = {
        'date_of_report': r'Date of report:\s*(\d{2}/\d{2}/\d{4})',
        'reference': r'Ref:\s*([\d-]+)',
        'deceased_name': r'Deceased name:\s*([^\n]+?)(?=\s*Coroner name:|$)',
        'coroner_name': r'Coroner name:\s*([^\n]+?)(?=\s*Coroner Area:|$)',
        'coroner_area': r'Coroner Area:\s*([^\n]+?)(?=\s*Category:|$)',
        'categories': r'Category:\s*([^\n]+?)(?=\s*This report is being sent to:|$)',
        'sent_to': r'This report is being sent to:\s*([^\n]+)'
    }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text to ensure consistent format"""
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Ensure each metadata field starts on a new line
        for field in ['Date of report:', 'Ref:', 'Deceased name:', 'Coroner name:', 
                     'Coroner Area:', 'Category:', 'This report is being sent to:']:
            text = text.replace(field, f'\n{field}')
        
        # Remove multiple spaces and normalize whitespace
        text = ' '.join(text.split())
        
        # Ensure proper spacing after colons
        text = re.sub(r':\s*', ': ', text)
        
        # Remove unnecessary Unicode characters
        text = re.sub(r'[\u200b\ufeff]', '', text)
        
        return text.strip()
    
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
        
        # Preprocess content
        processed_text = self._preprocess_text(content)
        
        # Extract each field using the defined patterns
        for field, pattern in self.METADATA_PATTERNS.items():
            match = re.search(pattern, processed_text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                if field == 'categories':
                    # Split categories on pipe and clean
                    categories = [cat.strip() for cat in value.split('|')]
                    metadata[field] = categories
                else:
                    metadata[field] = value
        
        return metadata

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe to extract metadata from content"""
    extractor = MetadataExtractor()
    metadata_rows = []
    
    for _, row in df.iterrows():
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
        
        # Try to extract from main content
        content_metadata = extractor.extract_metadata(row['Content'])
        metadata.update({k: v for k, v in content_metadata.items() if v})
        
        # Try to extract from PDF contents if available
        pdf_columns = [col for col in df.columns if col.startswith('PDF_') and col.endswith('_Content')]
        for pdf_col in pdf_columns:
            if pd.notna(row[pdf_col]):
                pdf_metadata = extractor.extract_metadata(row[pdf_col])
                # Update only if we find new information
                metadata.update({k: v for k, v in pdf_metadata.items() if v and not metadata[k]})
        
        metadata_rows.append(metadata)
    
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
