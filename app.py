import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from pyvis.network import Network
import math
import random
import tempfile
import base64
from pathlib import Path
from matplotlib import patches
from itertools import combinations
import requests
from io import StringIO

CSV_FILES = [
    'Females_45to64.csv',
    'Females_65plus.csv', 
    'Females_below45.csv',
    'Males_45to64.csv',
    'Males_65plus.csv',
    'Males_below45.csv'
]

def get_readable_filename(filename):
    if filename == 'Females_45to64.csv':
        return 'Females 45 to 64 years'
    elif filename == 'Females_65plus.csv':
        return 'Females 65 years and over'
    elif filename == 'Females_below45.csv':
        return 'Females below 45 years'
    elif filename == 'Males_45to64.csv':
        return 'Males 45 to 64 years'
    elif filename == 'Males_65plus.csv':
        return 'Males 65 years and over'
    elif filename == 'Males_below45.csv':
        return 'Males below 45 years'
    else:
        return filename

def load_and_process_data(input_file):
    """Load and process the selected CSV file"""
    try:
        github_url = f"https://raw.githubusercontent.com/gcosma/personalised_mltc/main/data/{input_file}"
        print(f"Attempting to load from URL: {github_url}")  # Debug print
        try:
            response = requests.get(github_url)
            response.raise_for_status()  # Raise an exception for bad status codes  
            print(f"Response status code: {response.status_code}")  # Debug print
            data = pd.read_csv(StringIO(response.text))
        except Exception as e:
            print(f"Error fetching data: {str(e)}")  # Debug print   
            raise

        total_patients = data['TotalPatientsInGroup'].iloc[0]

        # Determine gender and age group from filename  
        filename_lower = input_file.lower()
        
        if 'females' in filename_lower:
            gender = 'Female'
        elif 'males' in filename_lower: 
            gender = 'Male'
        else:
            gender = 'Unknown Gender'

        if 'below45' in filename_lower:
            age_group = '<45' 
        elif '45to64' in filename_lower:
            age_group = '45-64'
        elif '65plus' in filename_lower:
            age_group = '65+'
        else:
            age_group = 'Unknown Age Group'

        return data, total_patients, gender, age_group

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        print(f"Detailed error: {str(e)}")  # Debug print
        import traceback 
        print(traceback.format_exc())  # Print full traceback
        return None, None, None, None
def clear_session_state():
    """Clear all analysis results from session state when a new file is uploaded"""
    st.session_state.sensitivity_results = None
    st.session_state.network_html = None
    st.session_state.combinations_results = None
    st.session_state.combinations_fig = None
    st.session_state.personalized_html = None
    st.session_state.selected_conditions = []
    st.session_state.min_or = 2.0
    st.session_state.time_horizon = 5
    st.session_state.time_margin = 0.1
    st.session_state.min_frequency = None
    st.session_state.min_percentage = None
    st.session_state.unique_conditions = []


# Disease category mappings
condition_categories = {
    "Anaemia": "Blood",
    "Cardiac Arrhythmias": "Circulatory",
    "Coronary Heart Disease": "Circulatory",
    "Heart Failure": "Circulatory",
    "Hypertension": "Circulatory",
    "Peripheral Vascular Disease": "Circulatory",
    "Stroke": "Circulatory",
    "Barretts Oesophagus": "Digestive",
    "Chronic Constipation": "Digestive",
    "Chronic Diarrhoea": "Digestive",
    "Cirrhosis": "Digestive",
    "Dysphagia": "Digestive",
    "Inflammatory Bowel Disease": "Digestive",
    "Reflux Disorders": "Digestive",
    "Hearing Loss": "Ear",
    "Addisons Disease": "Endocrine",
    "Diabetes": "Endocrine",
    "Polycystic Ovary Syndrome": "Endocrine",
    "Thyroid Disorders": "Endocrine",
    "Visual Impairment": "Eye",
    "Chronic Kidney Disease": "Genitourinary",
    "Menopausal And Perimenopausal": "Genitourinary",
    "Menopausal and Perimenopausal": "Genitourinary",
    "Dementia": "Mental",
    "Mental Illness": "Mental",
    "Tourette": "Mental",
    "Chronic Arthritis": "Musculoskeletal",
    "Chronic Pain Conditions": "Musculoskeletal",
    "Osteoporosis": "Musculoskeletal",
    "Cancer": "Neoplasms",
    "Cerebral Palsy": "Nervous",
    "Epilepsy": "Nervous",
    "Insomnia": "Nervous",
    "Multiple Sclerosis": "Nervous",
    "Neuropathic Pain": "Nervous",
    "Parkinsons": "Nervous",
    "Bronchiectasis": "Respiratory",
    "Chronic Airway Diseases": "Respiratory",
    "Chronic Pneumonia": "Respiratory",
    "Interstitial Lung Disease": "Respiratory",
    "Psoriasis": "Skin"
}


SYSTEM_COLORS = {
    "Blood": "#DC143C",        # Crimson
    "Circulatory": "#FF4500",  # Orange Red
    "Digestive": "#32CD32",    # Lime Green
    "Ear": "#4169E1",         # Royal Blue
    "Endocrine": "#BA55D3",    # Medium Orchid
    "Eye": "#1E90FF",         # Dodger Blue (changed from teal)
    "Genitourinary": "#DAA520", # Goldenrod
    "Mental": "#8B4513",       # Saddle Brown
    "Musculoskeletal": "#4682B4", # Steel Blue
    "Neoplasms": "#800080",    # Purple
    "Nervous": "#FFD700",      # Gold
    "Respiratory": "#006400",   # Dark Green (changed from teal)
    "Skin": "#F08080",        # Light Coral
    "Other": "#808080"         # Gray
}


def parse_iqr(iqr_string):
    """Parse IQR string of format 'median [Q1-Q3]' into (median, q1, q3)"""
    try:
        median_str, iqr = iqr_string.split(' [')
        q1, q3 = iqr.strip(']').split('-')
        return float(median_str), float(q1), float(q3)
    except:
        return 0.0, 0.0, 0.0


@st.cache_data
def perform_sensitivity_analysis(data, top_n=5):
    """Perform sensitivity analysis with configurable number of top trajectories"""
    or_thresholds = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    results = []
    total_patients = data['TotalPatientsInGroup'].iloc[0]

    # Get top n patterns from full dataset first
    top_patterns = data.nlargest(top_n, 'OddsRatio')[
        ['ConditionA', 'ConditionB', 'OddsRatio', 'PairFrequency',
         'MedianDurationYearsWithIQR', 'DirectionalPercentage', 'Precedence']
    ].to_dict('records')

    for threshold in or_thresholds:
        filtered_data = data[data['OddsRatio'] >= threshold].copy()
        n_trajectories = len(filtered_data)

        total_pairs = filtered_data['PairFrequency'].sum()
        estimated_unique_patients = total_pairs / 2
        coverage = min((estimated_unique_patients / total_patients) * 100, 100.0)

        system_pairs = set()
        for _, row in filtered_data.iterrows():
            sys_a = condition_categories.get(row['ConditionA'], 'Other')
            sys_b = condition_categories.get(row['ConditionB'], 'Other')
            if sys_a != sys_b:
                system_pairs.add(tuple(sorted([sys_a, sys_b])))

        duration_stats = filtered_data['MedianDurationYearsWithIQR'].apply(parse_iqr)
        medians = [x[0] for x in duration_stats if x[0] > 0]
        q1s = [x[1] for x in duration_stats if x[1] > 0]
        q3s = [x[2] for x in duration_stats if x[2] > 0]

        results.append({
            'OR_Threshold': threshold,
            'Num_Trajectories': n_trajectories,
            'Coverage_Percent': round(coverage, 2),
            'System_Pairs': len(system_pairs),
            'Median_Duration': round(np.median(medians) if medians else 0, 2),
            'Q1_Duration': round(np.median(q1s) if q1s else 0, 2),
            'Q3_Duration': round(np.median(q3s) if q3s else 0, 2),
            'Top_Patterns': top_patterns
        })

    return pd.DataFrame(results)

def create_patient_count_legend(G):
    """Create a dynamic patient count legend based on actual data values.
    Formats the legend to match the specified style with exact patient ranges and percentages.
    
    Args:
        G: NetworkX graph containing edge data with pair_frequency attribute
        
    Returns:
        str: HTML string containing the formatted legend
    """
    try:
        # Extract pair frequencies from graph edges and convert to integers
        pair_frequencies = [int(data['pair_frequency']) for _, _, data in G.edges(data=True)]
        
        if not pair_frequencies:
            return """<div>No data available for legend</div>"""
        
        # Calculate percentiles and round to integers
        percentiles = np.percentile(pair_frequencies, [0, 20, 40, 60, 80, 100])
        percentiles = [int(round(p)) for p in percentiles]
        
        # Define the CSS styles for the legend
        legend_styles = """
            <style>
                .legend-container {
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background: white;
                    padding: 15px;
                    border: 1px solid #ccc;
                    font-family: Arial, sans-serif;
                    font-size: 12px;
                    z-index: 1000;
                    box-shadow: 2px 2px 4px rgba(0,0,0,0.1);
                }
                .legend-title {
                    font-weight: bold;
                    margin-bottom: 10px;
                    color: #333;
                }
                .legend-item {
                    margin: 8px 0;
                    display: flex;
                    align-items: center;
                }
                .legend-line {
                    width: 40px;
                    margin-right: 10px;
                    background-color: black;
                }
                .legend-text {
                    color: #333;
                }
            </style>
        """
        
        # Create the legend container
        legend_html = f"""
            {legend_styles}
            <div class="legend-container">
                <div class="legend-title">Patient Count Ranges</div>
        """
        
        # Add the first four ranges
        for i in range(4):
            line_thickness = i + 1
            legend_html += f"""
                <div class="legend-item">
                    <div class="legend-line" style="height: {line_thickness}px;"></div>
                    <div class="legend-text">
                        {percentiles[i]} ≤ Patients < {percentiles[i+1]} ({i*20}% - {(i+1)*20}%)
                    </div>
                </div>
            """
        
        # Add the final range (80%+)
        legend_html += f"""
                <div class="legend-item">
                    <div class="legend-line" style="height: 5px;"></div>
                    <div class="legend-text">
                        Patients ≥ {percentiles[4]} (80%+)
                    </div>
                </div>
            </div>
        """
        
        return legend_html
        
    except Exception as e:
        print(f"Error creating legend: {str(e)}")
        return """<div>Error creating legend</div>"""



@st.cache_data
def create_network_graph(data, patient_conditions, min_or, time_horizon=None, time_margin=None):
    """Create network graph matching the personalized analysis visualization with cohort-style edges."""
    # Initialize network with higher resolution settings
    net = Network(height="1200px", width="100%", bgcolor='white', font_color='black', directed=True)

    # Enhanced network options
    net.set_options("""
    {
        "nodes": {
            "font": {"size": 24, "strokeWidth": 2},
            "scaling": {"min": 20, "max": 50}
        },
        "edges": {
            "color": {"inherit": false},
            "font": {
                "size": 18,
                "strokeWidth": 2,
                "align": "middle",
                "background": "rgba(255, 255, 255, 0.8)"
            },
            "smooth": {
                "type": "continuous",
                "roundness": 0.2
            }
        },
        "physics": {
            "enabled": false,
            "barnesHut": {
                "gravitationalConstant": -4000,
                "centralGravity": 0.1,
                "springLength": 250,
                "springConstant": 0.03,
                "damping": 0.1,
                "avoidOverlap": 1
            },
            "minVelocity": 0.75,
            "stabilization": {
                "enabled": true,
                "iterations": 1000,
                "updateInterval": 25
            }
        }
    }
    """)

    # Apply initial OR filter
    filtered_data = data[data['OddsRatio'] >= min_or].copy()
    total_patients = data['TotalPatientsInGroup'].iloc[0]
    
    # Find all connected conditions and their relationships
    connected_conditions = set()
    relationships_to_show = []
    
    for condition_a in patient_conditions:
        condition_relationships = filtered_data[
            (filtered_data['ConditionA'] == condition_a) |
            (filtered_data['ConditionB'] == condition_a)
        ]
        
        if time_horizon is not None and time_margin is not None:
            condition_relationships = condition_relationships[
                condition_relationships['MedianDurationYearsWithIQR'].apply(
                    lambda x: parse_iqr(x)[0]) <= time_horizon * (1 + time_margin)
            ]
        
        for _, row in condition_relationships.iterrows():
            other_condition = (row['ConditionB'] if row['ConditionA'] == condition_a 
                             else row['ConditionA'])
            
            if other_condition not in patient_conditions:
                connected_conditions.add(other_condition)
                relationships_to_show.append(row)
    
    active_conditions = set(patient_conditions) | connected_conditions

    # Organize by system
    system_conditions = {}
    for condition in active_conditions:
        category = condition_categories.get(condition, "Other")
        if category not in system_conditions:
            system_conditions[category] = []
        system_conditions[category].append(condition)

    # Calculate positions
    active_categories = {condition_categories[cond] for cond in active_conditions 
                        if cond in condition_categories}
    angle_step = (2 * math.pi) / len(active_categories)
    radius = 500
    system_centers = {}

    for i, category in enumerate(sorted(active_categories)):
        angle = i * angle_step
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        system_centers[category] = (x, y)

    # Create body systems legend
    systems_legend_html = """
    <div style="position: absolute; top: 10px; right: 10px; background: white;
                padding: 10px; border: 1px solid #ddd; border-radius: 5px; z-index: 1000;">
        <h3 style="margin-top: 0; margin-bottom: 10px;">Legend</h3>
        <div style="margin-bottom: 10px;">
            <strong>Node Types:</strong><br>
            ★ Initial Condition<br>
            ○ Related Condition
        </div>
        <div>
            <strong>Body Systems:</strong><br>
    """

    for system, color in SYSTEM_COLORS.items():
        systems_legend_html += f"""
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <div style="width: 15px; height: 15px; background-color: {color}50;
                 border: 1px solid {color}; margin-right: 5px;"></div>
            <span>{system}</span>
        </div>
        """

    systems_legend_html += """
        </div>
        <div style="margin-top: 10px;">
            <strong>Edge Information:</strong><br>
            • Edge thickness indicates strength of relationship<br>
            • Arrow indicates typical progression direction<br>
            • Hover over edges for detailed statistics
        </div>
    </div>
    """

    # Add nodes
    for category, conditions in system_conditions.items():
        center_x, center_y = system_centers[category]
        sub_radius = radius / (len(conditions) + 1)
        
        for j, condition in enumerate(conditions):
            sub_angle = (j / len(conditions)) * (2 * math.pi)
            node_x = center_x + sub_radius * math.cos(sub_angle)
            node_y = center_y + sub_radius * math.sin(sub_angle)
            
            is_initial = condition in patient_conditions
            node_label = f"★ {condition}" if is_initial else condition
            node_size = 40 if is_initial else 30
            base_color = SYSTEM_COLORS[category]
            
            net.add_node(
                condition,
                label=node_label,
                title=f"{condition}\nSystem: {category}",
                size=node_size,
                x=node_x,
                y=node_y,
                color={'background': f"{base_color}50", 
                       'border': '#000000' if is_initial else base_color},
                physics=True,
                fixed=False
            )

    # Get frequency percentiles for edge widths
    if relationships_to_show:
        freqs = [row['PairFrequency'] for row in relationships_to_show]
        percentiles = np.percentile(freqs, [0, 20, 40, 60, 80, 100])
    else:
        percentiles = np.zeros(6)  # Default if no relationships

    # Create dynamic edge width legend
    edge_legend_html = """
    <div style="position: absolute; top: 10px; left: 10px; background: white;
                padding: 10px; border: 1px solid #ddd; border-radius: 5px; z-index: 1000;">
        <div style="font-weight: bold; margin-bottom: 5px;">Patient Count Ranges</div>
    """
    
    # Add ranges to legend
    ranges = [
        (percentiles[0], percentiles[1], "0% - 20%"),
        (percentiles[1], percentiles[2], "20% - 40%"),
        (percentiles[2], percentiles[3], "40% - 60%"),
        (percentiles[3], percentiles[4], "60% - 80%"),
        (percentiles[4], percentiles[5], "80%+")
    ]
    
    for i, (lower, upper, label) in enumerate(ranges, 1):
        edge_legend_html += f"""
        <div style="margin: 5px 0;">
            <div style="border-bottom: {i}px solid black; width: 40px; 
                       display: inline-block; margin-right: 5px;"></div>
            {int(lower)} ≤ Patients < {int(upper)} ({label})
        </div>
        """
    edge_legend_html += "</div>"

    # Add edges with standardized widths
    processed_edges = set()
    
    for row in relationships_to_show:
        condition_a = row['ConditionA']
        condition_b = row['ConditionB']
        
        edge_pair = tuple(sorted([condition_a, condition_b]))
        if edge_pair in processed_edges:
            continue
        processed_edges.add(edge_pair)

        if "precedes" in row['Precedence']:
            parts = row['Precedence'].split(" precedes ")
            source = parts[0]
            target = parts[1]
            percentage = (row['DirectionalPercentage'] 
                        if source == condition_a 
                        else (100 - row['DirectionalPercentage']))
        else:
            source = condition_a
            target = condition_b
            percentage = row['DirectionalPercentage']

        # Calculate edge width based on percentiles
        freq = row['PairFrequency']
        for i, (lower, upper, _) in enumerate(ranges, 1):
            if lower <= freq < upper:
                edge_width = i
                break
        else:
            edge_width = 5  # Maximum width for highest range

        prevalence = (row['PairFrequency'] / total_patients) * 100
        
        edge_label = (
            f"OR: {row['OddsRatio']:.1f}\n"
            f"Years: {row['MedianDurationYearsWithIQR']}\n"
            f"n={row['PairFrequency']} ({prevalence:.1f}%)\n"
            f"Proceeds: {percentage:.1f}%"
        )

        net.add_edge(
            source,
            target,
            label=edge_label,
            title=edge_label,
            width=edge_width,
            arrows={'to': {'enabled': True, 'scaleFactor': 1}},
            color={'color': 'rgba(128,128,128,0.7)', 'highlight': 'black'},
            smooth={'type': 'curvedCW', 'roundness': 0.2}
        )

    # Add export and download functionality
    export_script = """
    <script type="text/javascript">
    function exportHighRes() {
        const network = document.getElementsByTagName('canvas')[0];
        const scale = 3;
        
        const exportCanvas = document.createElement('canvas');
        exportCanvas.width = network.width * scale;
        exportCanvas.height = network.height * scale;
        
        const ctx = exportCanvas.getContext('2d');
        ctx.scale(scale, scale);
        ctx.drawImage(network, 0, 0);
        
        const link = document.createElement('a');
        link.download = 'trajectory_network.png';
        link.href = exportCanvas.toDataURL('image/png');
        link.click();
    }

    function downloadNetwork() {
        const htmlContent = document.documentElement.outerHTML;
        const blob = new Blob([htmlContent], { type: 'text/html' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'network.html';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    }
    </script>
    """

    buttons_html = """
    <div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;
                display: flex; gap: 10px;">
        <button onclick="exportHighRes()" 
                style="padding: 10px 20px; font-size: 16px; background-color: #4CAF50;
                       color: white; border: none; border-radius: 5px; cursor: pointer;">
            📸 Download High-Res Image
        </button>
        <button onclick="downloadNetwork()" 
                style="padding: 10px 20px; font-size: 16px; background-color: #4CAF50;
                       color: white; border: none; border-radius: 5px; cursor: pointer;">
            📥 Download Network
        </button>
    </div>
    """

    # Generate final HTML with all components
    network_html = net.generate_html()
    final_html = network_html.replace(
        '</head>',
        export_script + '</head>'
    ).replace(
        '</body>',
        f'{systems_legend_html}{edge_legend_html}{buttons_html}</body>'
    )

    return final_html

@st.cache_data
def analyze_condition_combinations(data, min_percentage, min_frequency):
    """Analyze combinations of conditions"""
    total_patients = data['TotalPatientsInGroup'].iloc[0]

    filtered_data = data[
        (data['Percentage'] >= min_percentage) &
        (data['PairFrequency'] >= min_frequency)
    ].copy()

    # Clean condition names
    for col in ['ConditionA', 'ConditionB']:
        filtered_data[col] = (filtered_data[col]
                            .str.replace(r'\s*\([^)]*\)', '', regex=True)
                            .str.replace('_', ' '))

    unique_conditions = pd.unique(filtered_data[['ConditionA', 'ConditionB']].values.ravel('K'))

    # Calculate frequencies
    pair_frequency_map = {}
    condition_frequency_map = {}

    for _, row in filtered_data.iterrows():
        for key in [f"{row['ConditionA']}_{row['ConditionB']}",
                   f"{row['ConditionB']}_{row['ConditionA']}"]:
            pair_frequency_map[key] = row['PairFrequency']

        for condition in [row['ConditionA'], row['ConditionB']]:
            condition_frequency_map[condition] = (
                condition_frequency_map.get(condition, 0) + row['PairFrequency']
            )

    # Analyze combinations
    result_data = []
    for k in range(3, min(8, len(unique_conditions) + 1)):
        for comb in combinations(unique_conditions, k):
            pair_frequencies = [
                pair_frequency_map.get(f"{a}_{b}", 0)
                for a, b in combinations(comb, 2)
            ]

            frequency = min(pair_frequencies)
            prevalence = (frequency / total_patients) * 100

            # Calculate odds ratio
            observed = frequency
            expected = total_patients
            for condition in comb:
                expected *= (condition_frequency_map[condition] / total_patients)

            odds_ratio = observed / expected if expected != 0 else float('inf')

            result_data.append({
                'Combination': ' + '.join(comb),
                'NumConditions': len(comb),
                'Minimum Pair Frequency': frequency,
                'Prevalence of the combination (%)': prevalence,
                'Total odds ratio': odds_ratio
            })

    results_df = pd.DataFrame(result_data)
    results_df = (results_df[results_df['Prevalence of the combination (%)'] > 0]
                 .sort_values('Prevalence of the combination (%)', ascending=False))

    return results_df

def create_sensitivity_plot(results):
    """Create the sensitivity analysis visualization"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    x_vals = results['OR_Threshold'].values
    bar_heights = results['Num_Trajectories']

    # Plot bars and lines
    bars = ax1.bar(x_vals, bar_heights, alpha=0.3, color='navy')
    line = ax2.plot(x_vals, results['Coverage_Percent'], 'r-o', linewidth=2)

    # Add scatter plot with variable sizes
    sizes = (results['System_Pairs'] / results['System_Pairs'].max()) * 500
    scatter = ax2.scatter(x_vals, results['Coverage_Percent'], s=sizes, alpha=0.5, color='darkred')

    # Add text annotations
    for i, row in results.iterrows():
        ax1.text(row['OR_Threshold'], bar_heights[i] * 0.5,
                f"Median: {row['Median_Duration']:.1f}y\nIQR: [{row['Q1_Duration']:.1f}-{row['Q3_Duration']:.1f}]",
                ha='center', va='center', fontsize=10)

    # Labels and legend
    ax1.set_xlabel('Minimum Odds Ratio Threshold')
    ax1.set_ylabel('Number of Disease Trajectories')
    ax2.set_ylabel('Population Coverage (%)')

    legend_elements = [
        patches.Patch(facecolor='navy', alpha=0.3, label='Number of Trajectories'),
        Line2D([0], [0], color='r', marker='o', label='Population Coverage %'),
        Line2D([0], [0], marker='o', color='darkred', alpha=0.5,
               label='System Pairs', markersize=10, linestyle='None')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    plt.title('Impact of Odds Ratio Threshold on Disease Trajectory Analysis')
    plt.tight_layout()
    return fig

def create_combinations_plot(results_df):
    """Create the combinations analysis visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))

    top_10 = results_df.nlargest(10, 'Prevalence of the combination (%)')
    bars = ax.bar(range(len(top_10)), top_10['Prevalence of the combination (%)'])

    # Customize the plot
    ax.set_xticks(range(len(top_10)))
    ax.set_xticklabels(top_10['Combination'], rotation=45, ha='right')
    ax.set_title('Top 10 Condition Combinations by Prevalence')
    ax.set_xlabel('Condition Combinations')
    ax.set_ylabel('Prevalence (%)')

    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')

    plt.tight_layout()
    return fig

# Add this function near the other functions, before main():
def create_personalized_analysis(data, patient_conditions, time_horizon=None, time_margin=None, min_or=2.0):
    """Create a personalized analysis of disease trajectories for a patient's conditions"""
    filtered_data = data[data['OddsRatio'] >= min_or].copy()
    total_patients = data['TotalPatientsInGroup'].iloc[0]

    def get_risk_level(odds_ratio):
        if odds_ratio >= 5:
            return "High", "#dc3545"
        elif odds_ratio >= 3:
            return "Moderate", "#ffc107"
        else:
            return "Low", "#28a745"

    html = """
    <style>
        .patient-analysis {
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont;
            margin: 20px 0;
            width: 100%;
            max-width: 100%;
        }
        .condition-section {
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background-color: #f8f9fa;
            width: 100%;
        }
        .condition-header {
            font-size: 1.2em;
            color: #2c5282;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e2e8f0;
        }
        .trajectory-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            background-color: white;
            font-size: 14px;
        }
        .trajectory-table th {
            background-color: #f5f5f5;
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
            white-space: nowrap;
        }
        .trajectory-table td {
            padding: 10px;
            border: 1px solid #ddd;
            vertical-align: top;
        }
        .risk-badge {
            padding: 4px 8px;
            border-radius: 4px;
            color: white;
            font-weight: bold;
        }
        .system-tag {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            background-color: #e2e8f0;
            font-size: 0.9em;
            margin-right: 5px;
        }
        .timeline-indicator {
            font-style: italic;
            color: #666;
        }
        .progression-arrow {
            color: #4a5568;
            font-weight: bold;
        }
        .percentage {
            color: #2d3748;
            font-weight: bold;
        }
        @media (max-width: 1200px) {
            .trajectory-table {
                display: block;
                overflow-x: auto;
                white-space: nowrap;
            }
        }
        .analysis-container {
            max-width: 100%;
            margin: 0 auto;
            padding: 20px;
        }
        .summary-section {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid #e2e8f0;
        }
    </style>
    <div class="patient-analysis">
        <div class="analysis-container">
            <h2>Personalized Disease Trajectory Analysis</h2>
            <div class="summary-section">
                <h3>Current Conditions:</h3>
                <p>""" + ", ".join(f"<span class='system-tag'>{condition_categories.get(cond, 'Other')}</span> {cond}" for cond in patient_conditions) + """</p>
            </div>
    """

    for condition_a in patient_conditions:
        time_filtered_data = filtered_data[
            (filtered_data['ConditionA'] == condition_a) |
            (filtered_data['ConditionB'] == condition_a)
        ]

        if time_horizon and time_margin:
            time_filtered_data = time_filtered_data[
                time_filtered_data['MedianDurationYearsWithIQR'].apply(
                    lambda x: parse_iqr(x)[0]) <= time_horizon * (1 + time_margin)
            ]

        if not time_filtered_data.empty:
            system_a = condition_categories.get(condition_a, 'Other')
            html += f"""
            <div class="condition-section">
                <div class="condition-header">
                    <span class="system-tag">{system_a}</span>
                    Progression Paths from {condition_a}
                </div>
                <table class="trajectory-table">
                    <thead>
                        <tr>
                            <th>Risk Level</th>
                            <th>Potential Progression</th>
                            <th>Expected Timeline</th>
                            <th>Statistical Support</th>
                            <th>Progression Details</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for _, row in time_filtered_data.sort_values('OddsRatio', ascending=False).iterrows():
                if row['ConditionA'] == condition_a:
                    other_condition = row['ConditionB']
                    direction_percentage = row['DirectionalPercentage']
                else:
                    other_condition = row['ConditionA']
                    direction_percentage = 100 - row['DirectionalPercentage']

                if other_condition not in patient_conditions:
                    system_b = condition_categories.get(other_condition, 'Other')
                    median, q1, q3 = parse_iqr(row['MedianDurationYearsWithIQR'])
                    prevalence = (row['PairFrequency'] / total_patients) * 100
                    risk_level, color = get_risk_level(row['OddsRatio'])

                    # Parse precedence to determine direction
                    if "precedes" in row['Precedence']:
                        parts = row['Precedence'].split(" precedes ")
                        first_condition = parts[0]
                        second_condition = parts[1]
                        direction = f"{first_condition} <span class='progression-arrow'>→</span> {second_condition}"
                        if first_condition == row['ConditionA']:
                            percentage = row['DirectionalPercentage']
                        else:
                            percentage = 100 - row['DirectionalPercentage']

                        progression_text = f"""
                            {direction}<br>
                            <span class='percentage'>{percentage:.1f}%</span> of cases follow this pattern
                        """
                    else:
                        direction = f"{condition_a} <span class='progression-arrow'>→</span> {other_condition}"
                        progression_text = f"""
                            {direction}<br>
                            <span class='percentage'>{direction_percentage:.1f}%</span> of cases follow this pattern
                        """

                    html += f"""
                        <tr>
                            <td><span class="risk-badge" style="background-color: {color}">{risk_level}</span></td>
                            <td>
                                <strong>{other_condition}</strong><br>
                                <span class="system-tag">{system_b}</span>
                            </td>
                            <td class="timeline-indicator">
                                Typically {median:.1f} years<br>
                                Range: {q1:.1f} to {q3:.1f} years
                            </td>
                            <td>
                                OR: {row['OddsRatio']:.1f}<br>
                                {row['PairFrequency']} cases ({prevalence:.1f}%)
                            </td>
                            <td>
                                {progression_text}
                            </td>
                        </tr>
                    """

            html += """
                    </tbody>
                </table>
            </div>
            """

    html += """
            <div class="summary-section">
                <h4>Understanding This Analysis:</h4>
                <ul>
                    <li><strong>Risk Level:</strong> Based on odds ratio strength (High: OR≥5, Moderate: OR≥3, Low: OR≥2)</li>
                    <li><strong>Expected Timeline:</strong> Median years and range between which progression typically occurs</li>
                    <li><strong>Statistical Support:</strong> Odds ratio and number of observed cases in the population</li>
                    <li><strong>Progression Details:</strong> Direction of progression and percentage of cases that follow this pattern</li>
                </ul>
            </div>
        </div>
    </div>
    """

    return html


    
def create_network_visualization(data, min_or, min_freq):
    """Create network visualization with legends with pastel colors matching paper style"""
    net = Network(height="800px", width="100%", bgcolor='white', font_color='black', directed=True)
    
    # Filter data
    filtered_data = data[
        (data['OddsRatio'] >= min_or) &
        (data['PairFrequency'] >= min_freq)
    ].copy()

    # Create condition categories legend HTML - Note the color50 for pastel effect
    legend_html = """
    <div style="position: absolute; top: 10px; left: 10px; background: white; padding: 10px; 
                border: 1px solid #ccc; font-size: 12px;">
        <div style="font-weight: bold; margin-bottom: 5px;">Condition Categories</div>
    """
    for category, color in sorted(SYSTEM_COLORS.items()):
        if category != "Other":
            legend_html += f"""
            <div style="margin: 2px 0;">
                <div style="display: inline-block; width: 20px; height: 20px; background: {color}50;
                     border: 1px solid {color}; margin-right: 5px;"></div>
                {category}
            </div>
            """
    legend_html += "</div>"

    # Create patient count ranges legend
    count_legend = """
    <div style="position: absolute; top: 10px; right: 10px; background: white; padding: 10px; 
                border: 1px solid #ccc; font-size: 12px;">
        <div style="font-weight: bold; margin-bottom: 5px;">Patient Count Ranges</div>
    """
    
    # Get frequency percentiles for edge widths
    freqs = filtered_data['PairFrequency'].values
    percentiles = np.percentile(freqs, [0, 20, 40, 60, 80, 100])
    
    # Add ranges to legend
    ranges = [
        (percentiles[0], percentiles[1], "0% - 20%"),
        (percentiles[1], percentiles[2], "20% - 40%"),
        (percentiles[2], percentiles[3], "40% - 60%"),
        (percentiles[3], percentiles[4], "60% - 80%"),
        (percentiles[4], percentiles[5], "80%+")
    ]
    
    for i, (lower, upper, label) in enumerate(ranges, 1):
        count_legend += f"""
        <div style="margin: 5px 0;">
            <div style="border-bottom: {i}px solid black; width: 40px; display: inline-block; margin-right: 5px;"></div>
            {int(lower)} ≤ Patients < {int(upper)} ({label})
        </div>
        """
    count_legend += "</div>"

    # Network options for clear visualization
    net.set_options("""
    {
        "nodes": {
            "font": {"size": 14},
            "shape": "dot"
        },
        "edges": {
            "font": {
                "size": 8,
                "align": "middle",
                "background": "white"
            },
            "smooth": {"type": "curvedCW", "roundness": 0.2}
        },
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 200
            }
        }
    }
    """)

    # Add nodes with system-based layout and pastel colors
    unique_systems = set(condition_categories[cond] for cond in set(filtered_data['ConditionA']) | set(filtered_data['ConditionB']))
    radius = 300
    system_angles = {sys: i * (2 * math.pi / len(unique_systems)) for i, sys in enumerate(sorted(unique_systems))}
    
    # Add nodes with pastel colors
    for condition in set(filtered_data['ConditionA']) | set(filtered_data['ConditionB']):
        system = condition_categories.get(condition, "Other")
        base_color = SYSTEM_COLORS.get(system, SYSTEM_COLORS["Other"])
        angle = system_angles[system]
        
        # Add random variation to position
        x = radius * math.cos(angle) + random.uniform(-50, 50)
        y = radius * math.sin(angle) + random.uniform(-50, 50)
        
        # Create node with pastel color (using 50% transparency)
        net.add_node(
            condition,
            label=condition,
            title=f"{condition}\nSystem: {system}",
            x=x,
            y=y,
            color={'background': f"{base_color}50", 'border': base_color},
            size=30
        )

    # Add edges
    for _, row in filtered_data.iterrows():
        freq = row['PairFrequency']
        
        # Determine edge width based on frequency percentiles
        if freq < percentiles[1]:
            width = 1
        elif freq < percentiles[2]:
            width = 2
        elif freq < percentiles[3]:
            width = 3
        elif freq < percentiles[4]:
            width = 4
        else:
            width = 5

        # Edge label showing OR and Years
        edge_label = f"OR: {row['OddsRatio']:.1f}\nYears: {row['MedianDurationYearsWithIQR']}"
        
        net.add_edge(
            row['ConditionA'],
            row['ConditionB'],
            label=edge_label,
            title=edge_label,
            width=width,
            arrows={'to': {'enabled': True, 'scaleFactor': 0.5}},
            color={'color': 'rgba(128,128,128,0.7)', 'highlight': 'black'},
            font={'size': 8, 'color': 'black', 'strokeWidth': 2, 'strokeColor': 'white'}
        )

    # Generate final HTML with legends
    html = net.generate_html()
    final_html = html.replace('</body>', f'{legend_html}{count_legend}</body>')
    
    return final_html

def add_cohort_tab():
    """Add cohort analysis tab to the main app"""
    st.header("Cohort Network Analysis")
    st.markdown("""
    Visualize relationships between conditions as a network graph. 
    Node colors represent body systems, and edge thickness indicates association strength.
    """)

    main_col, control_col = st.columns([3, 1])

    with control_col:
        with st.container():
            st.markdown("### Control Panel")
            
            # Sliders for filtering
            min_or = st.slider(
                "Minimum Odds Ratio",
                1.0, 15.0, 2.0, 0.1,
                key="cohort_min_or",
                help="Filter relationships by minimum odds ratio"
            )

            min_freq = st.slider(
                "Minimum Pair Frequency",
                int(data['PairFrequency'].min()),
                int(data['PairFrequency'].max()),
                int(data['PairFrequency'].min()),
                help="Minimum number of occurrences required"
            )

            generate_button = st.button(
                "🔄 Generate Network",
                key="generate_network",
                help="Create network visualization"
            )

    with main_col:
        if generate_button:
            with st.spinner("🌐 Generating network visualization..."):
                try:
                    # Create visualization
                    html_content = create_network_visualization(data, min_or, min_freq)
                    
                    # Display network
                    st.components.v1.html(html_content, height=800)
                    
                    # Add download button
                    st.download_button(
                        label="📥 Download Network Visualization",
                        data=html_content,
                        file_name="condition_network.html",
                        mime="text/html"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating network visualization: {str(e)}")


# Add these imports at the top with your other imports
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # First run or password not correct
    if "password_correct" not in st.session_state:
        # Show input for password
        st.text_input(
            "Please enter the password",
            type="password",
            key="password",
            on_change=password_entered
        )
        return False

    # Password correct
    elif st.session_state["password_correct"]:
        return True
        
def add_footer():
    st.markdown(
        """
        <div class="footer">
            <div class="footer-copyright">
                <p>© 2024 DECODE Project. Loughborough University. Funded by the National Institute for Health and Care Research (NIHR). <a href="https://decode-project.org/research/"> DECODE Project Website </a> </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )  


def main():
    # Initialize session state for data persistence
    if 'sensitivity_results' not in st.session_state:
        st.session_state.sensitivity_results = None
    if 'network_html' not in st.session_state:
        st.session_state.network_html = None
    if 'combinations_results' not in st.session_state:
        st.session_state.combinations_results = None
    if 'combinations_fig' not in st.session_state:
        st.session_state.combinations_fig = None
    if 'personalized_html' not in st.session_state:
        st.session_state.personalized_html = None
    if 'selected_conditions' not in st.session_state:
        st.session_state.selected_conditions = []
    if 'min_or' not in st.session_state:
        st.session_state.min_or = 2.0
    if 'time_horizon' not in st.session_state:
        st.session_state.time_horizon = 5
    if 'time_margin' not in st.session_state:
        st.session_state.time_margin = 0.1
    if 'min_frequency' not in st.session_state:
        st.session_state.min_frequency = None
    if 'min_percentage' not in st.session_state:
        st.session_state.min_percentage = None
    if 'unique_conditions' not in st.session_state:
        st.session_state.unique_conditions = []
    if 'data_hash' not in st.session_state:
        st.session_state.data_hash = None
    if 'top_n_trajectories' not in st.session_state:
        st.session_state.top_n_trajectories = 5

    # Page configuration
    st.set_page_config(
        page_title="DECODE Project: Multimorbidity Analysis Tool for people with learning disability and MLTCs",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Check password before showing any content
    if not check_password():
        st.stop()

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton > button {
            width: 100%;
            height: 3rem;
            margin: 1rem 0;
            background-color: #ff4b4b;
            color: white;
        }
        .stButton > button:hover {
            background-color: #ff3333;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            padding: 1rem 0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 4rem;
            white-space: pre-wrap;
            background-color: #e2e8f0;
            color: #1a202c;
            border-radius: 4px;
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #cbd5e0;
            color: #1a202c;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #2c5282;
            color: white;
        }        
        div[data-testid="stSidebarNav"] {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
        }
        div[data-testid="stFileUploader"] {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 4px;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🏥 DECODE: Multimorbidity Analysis Tool for people with learning disability and MLTCs")
    st.markdown("""
    This tool helps analyse disease trajectories and comorbidity patterns in patient data.
    Upload your data file to begin analysis.
    """)

    try:
        # File selection container
        with st.container():
            selected_file = st.selectbox(
                "Select dataset", 
                CSV_FILES,
                format_func=get_readable_filename, 
                help="Choose from available datasets"
            )
            
        if selected_file:
            try:
                # Calculate hash of selected file  
                current_hash = hash(selected_file)
                
                # Check if this is a new file
                if 'data_hash' not in st.session_state or current_hash != st.session_state.data_hash:
                    # Clear all session state variables 
                    clear_session_state()
                    # Update the hash
                    st.session_state.data_hash = current_hash
                
                # Load and process data
                data, total_patients, gender, age_group = load_and_process_data(selected_file)
                
                if data is None:
                    st.error("Failed to load data. Please check your file selection.")
                    st.stop()
                
                # Data summary in an info box
                st.info(f"""
                📊 **Data Summary**  
                - Total Patients: {total_patients:,}
                - Gender: {gender}
                - Age Group: {age_group}
                """)



                # Create tabs with icons
                tabs = st.tabs([
                    "📈 Sensitivity Analysis",
                    "🔍 Condition Combinations",
                    "👤 Personalised Analysis",
                    "🎯 Personalised Trajectory Filter",
                    "🌐 Cohort Network"  
                ])

                # Sensitivity Analysis Tab
                with tabs[0]:
                    st.header("Sensitivity Analysis")
                    st.markdown("""
                    Explore how different odds ratio thresholds affect the number of disease
                    trajectories and population coverage.
                    """)

                    main_col, control_col = st.columns([3, 1])

                    with control_col:
                        with st.container():
                            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
                            st.markdown("### Control Panel")
                            top_n = st.slider(
                                "Number of Top Trajectories",
                                min_value=1,
                                max_value=20,
                                value=st.session_state.top_n_trajectories,
                                step=1,
                                help="Select how many top trajectories to display"
                            )
                            st.session_state.top_n_trajectories = top_n
                            
                            analyse_button = st.button(
                                "🚀 Run Analysis",
                                key="run_sensitivity",
                                help="Click to perform sensitivity analysis"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

                    with main_col:
                        try:
                            if analyse_button:
                                with st.spinner("💫 Analysing data..."):
                                    # Clear previous results
                                    st.session_state.sensitivity_results = None
                                    
                                    # Generate new results with selected top_n
                                    results = perform_sensitivity_analysis(data, top_n=top_n)
                                    st.session_state.sensitivity_results = results

                                    display_df = results.drop('Top_Patterns', axis=1)
                                    st.subheader("Analysis Results")
                                    st.dataframe(
                                        display_df.style.background_gradient(cmap='YlOrRd', subset=['Coverage_Percent'])
                                    )

                                    st.subheader(f"Top {top_n} Strongest Trajectories")
                                    patterns_df = pd.DataFrame(results.iloc[0]['Top_Patterns'])
                                    st.dataframe(
                                        patterns_df.style.background_gradient(cmap='YlOrRd', subset=['OddsRatio'])
                                    )

                                    fig = create_sensitivity_plot(results)
                                    st.pyplot(fig)

                                    csv = display_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Results",
                                        data=csv,
                                        file_name="sensitivity_analysis_results.csv",
                                        mime="text/csv"
                                    )
                            
                            # Display existing results if available
                            elif st.session_state.sensitivity_results is not None:
                                results = st.session_state.sensitivity_results
                                display_df = results.drop('Top_Patterns', axis=1)
                                
                                st.subheader("Analysis Results")
                                st.dataframe(
                                    display_df.style.background_gradient(cmap='YlOrRd', subset=['Coverage_Percent'])
                                )

                                num_patterns = len(results.iloc[0]['Top_Patterns'])
                                st.subheader(f"Top {num_patterns} Strongest Trajectories")
                                patterns_df = pd.DataFrame(results.iloc[0]['Top_Patterns'])
                                st.dataframe(
                                    patterns_df.style.background_gradient(cmap='YlOrRd', subset=['OddsRatio'])
                                )

                                fig = create_sensitivity_plot(results)
                                st.pyplot(fig)

                                csv = display_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results",
                                    data=csv,
                                    file_name="sensitivity_analysis_results.csv",
                                    mime="text/csv"
                                )

                        except Exception as e:
                            st.error(f"Error in sensitivity analysis: {str(e)}")
                            st.session_state.sensitivity_results = None

                # Condition Combinations Tab
                with tabs[1]:
                    st.header("Condition Combinations Analysis")
                    
                    main_col, control_col = st.columns([3, 1])

                    with control_col:
                        with st.container():
                            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
                            st.markdown("### Control Panel")
                            try:
                                min_freq_range = (data['PairFrequency'].min(), data['PairFrequency'].max())
                                min_frequency = st.slider(
                                    "Minimum Pair Frequency",
                                    int(min_freq_range[0]),
                                    int(min_freq_range[1]),
                                    int(min_freq_range[0]) if st.session_state.min_frequency is None
                                    else st.session_state.min_frequency,
                                    help="Minimum number of occurrences required"
                                )
                                st.session_state.min_frequency = min_frequency

                                min_percentage_range = (data['Percentage'].min(), data['Percentage'].max())
                                min_percentage = st.slider(
                                    "Minimum Prevalence (%)",
                                    float(min_percentage_range[0]),
                                    float(min_percentage_range[1]),
                                    float(min_percentage_range[0]) if st.session_state.min_percentage is None
                                    else st.session_state.min_percentage,
                                    0.1,
                                    help="Minimum percentage of population affected"
                                )
                                st.session_state.min_percentage = min_percentage

                                analyse_combinations_button = st.button(
                                    "🔍 Analyse Combinations",
                                    help="Click to analyse condition combinations"
                                )
                            except Exception as e:
                                st.error(f"Error setting up analysis parameters: {str(e)}")
                            st.markdown('</div>', unsafe_allow_html=True)

                    with main_col:
                        try:
                            if analyse_combinations_button:
                                with st.spinner("🔄 Analysing combinations..."):
                                    # Clear previous results
                                    st.session_state.combinations_results = None
                                    st.session_state.combinations_fig = None
                                    
                                    # Generate new results
                                    results_df = analyze_condition_combinations(
                                        data,
                                        min_percentage,
                                        min_frequency
                                    )

                                    if not results_df.empty:
                                        st.session_state.combinations_results = results_df
                                        
                                        st.subheader(f"Analysis Results ({len(results_df)} combinations)")
                                        st.dataframe(
                                            results_df.style.background_gradient(
                                                cmap='YlOrRd',
                                                subset=['Prevalence of the combination (%)']
                                            )
                                        )

                                        fig = create_combinations_plot(results_df)
                                        st.session_state.combinations_fig = fig
                                        st.pyplot(fig)

                                        csv = results_df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Results",
                                            data=csv,
                                            file_name="condition_combinations.csv",
                                            mime="text/csv"
                                        )
                                    else:
                                        st.warning("No combinations found matching the criteria. Try adjusting the parameters.")
                            
                            # Display existing results if available
                            elif st.session_state.combinations_results is not None:
                                results_df = st.session_state.combinations_results
                                st.subheader(f"Analysis Results ({len(results_df)} combinations)")
                                st.dataframe(
                                    results_df.style.background_gradient(
                                        cmap='YlOrRd',
                                        subset=['Prevalence of the combination (%)']
                                    )
                                )

                                if st.session_state.combinations_fig is not None:
                                    st.pyplot(st.session_state.combinations_fig)

                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results",
                                    data=csv,
                                    file_name="condition_combinations.csv",
                                    mime="text/csv"
                                )

                        except Exception as e:
                            st.error(f"Error in combinations analysis: {str(e)}")
                            st.session_state.combinations_results = None
                            st.session_state.combinations_fig = None

                # Personalised Analysis Tab
                with tabs[2]:
                    st.header("Personalised Trajectory Analysis")
                    st.markdown("""
                    Analyse potential disease progressions based on a patient's current conditions,
                    considering population-level statistics and time-based progression patterns.
                    """)
                
                    main_col, control_col = st.columns([3, 1])
                
                    with control_col:
                        with st.container():
                            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
                            st.markdown("### Control Panel")
                            
                            # Get min/max values from data
                            min_or_value = float(data['OddsRatio'].min())
                            max_or_value = float(data['OddsRatio'].max())
                            
                            min_or = st.slider(
                                "Minimum Odds Ratio",
                                min_value=min_or_value,
                                max_value=max_or_value,
                                value=st.session_state.min_or,
                                step=0.5,
                                key="personal_min_or",
                                help="Filter trajectories by minimum odds ratio"
                            )
                            st.session_state.min_or = min_or



                        
                                                                
                            # Get max years from data
                            max_years = math.ceil(data['MedianDurationYearsWithIQR'].apply(
                                lambda x: parse_iqr(x)[0]).max())
                            
                            time_horizon = st.slider(
                                "Time Horizon (years)",
                                min_value=1.0,
                                max_value=float(max_years),
                                value=float(st.session_state.time_horizon),
                                step=0.5,
                                key="personal_time_horizon",
                                help="Maximum time period to consider"
                            )
                            st.session_state.time_horizon = time_horizon
                    
                            time_margin = st.slider(
                                "Time Margin",
                                min_value=0.0,
                                max_value=0.5,
                                value=st.session_state.time_margin,
                                step=0.05,
                                key="personal_time_margin",
                                help="Allowable variation in time predictions"
                            )
                            st.session_state.time_margin = time_margin
                            
                            analyse_button = st.button(
                                "🔍 Analyse Trajectories",
                                key="personal_analyse",
                                help="Generate personalised analysis"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                
                    with main_col:
                        st.markdown("""
                        <h4 style='font-size: 1.2em; font-weight: 600; color: #333; margin-bottom: 10px;'>
                            🔍 Please select all conditions that the patient currently has:
                        </h4>
                        """, unsafe_allow_html=True)
                        
                        # Initialize session state for selected conditions if not exists
                        if 'selected_conditions' not in st.session_state:
                            st.session_state.selected_conditions = []
                
                        # Get unique conditions only once
                        unique_conditions = sorted(set(data['ConditionA'].unique()) | set(data['ConditionB'].unique()))
                        
                        def on_condition_select():
                            # Update the session state directly from the widget value
                            st.session_state.selected_conditions = st.session_state.personal_select
                
                        # Use the multiselect with a callback
                        selected_conditions = st.multiselect(
                            "Select Current Conditions",
                            options=unique_conditions,
                            default=st.session_state.selected_conditions,
                            key="personal_select",
                            on_change=on_condition_select,
                            help="Choose all conditions that the patient currently has"
                        )
                
                        if selected_conditions and analyse_button:
                            with st.spinner("🔄 Generating personalised analysis..."):
                                # Generate new analysis
                                html_content = create_personalized_analysis(
                                    data,
                                    selected_conditions,
                                    time_horizon,
                                    time_margin,
                                    min_or
                                )
                                st.session_state.personalized_html = html_content
                
                                html_container = f"""
                                <div style="min-height: 800px; width: 100%; padding: 20px;">
                                    {html_content}
                                </div>
                                """
                                st.components.v1.html(html_container, height=1200, scrolling=True)
                                st.download_button(
                                    label="📥 Download Analysis",
                                    data=html_content,
                                    file_name="personalised_trajectory_analysis.html",
                                    mime="text/html"
                                )
                
                        # Display existing analysis if available
                        elif st.session_state.personalized_html is not None:
                            html_container = f"""
                            <div style="min-height: 800px; width: 100%; padding: 20px;">
                                {st.session_state.personalized_html}
                            </div>
                            """
                            st.components.v1.html(html_container, height=1200, scrolling=True)
                            st.download_button(
                                label="📥 Download Analysis",
                                data=st.session_state.personalized_html,
                                file_name="personalised_trajectory_analysis.html",
                                mime="text/html"
                            )

        
        
                with tabs[3]:
                    st.header("Custom Trajectory Filter")
                    st.markdown("""
                    Visualise disease trajectories based on custom odds ratio and frequency thresholds.
                    Select conditions and adjust filters to explore different trajectory patterns.
                    """)
                
                    main_col, control_col = st.columns([3, 1])
                
                    with control_col:
                        with st.container():
                            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
                            st.markdown("### Control Panel")
                            try:
                                # Get min/max values from data
                                min_or_value = float(data['OddsRatio'].min())
                                max_or_value = float(data['OddsRatio'].max())
                                min_freq_value = int(data['PairFrequency'].min())
                                max_freq_value = int(data['PairFrequency'].max())
                                
                                min_or = st.slider(
                                    "Minimum Odds Ratio",
                                    min_value=min_or_value,
                                    max_value=max_or_value,
                                    value=st.session_state.min_or,
                                    step=0.5,
                                    key="custom_min_or",
                                    help="Filter trajectories by minimum odds ratio"
                                )
                
                                min_freq = st.slider(
                                    "Minimum Frequency",
                                    min_value=min_freq_value,
                                    max_value=max_freq_value,
                                    value=min_freq_value,
                                    step=1,
                                    help="Minimum number of occurrences required"
                                )
                
                                # Filter data based on both OR and frequency
                                filtered_data = data[
                                    (data['OddsRatio'] >= min_or) &
                                    (data['PairFrequency'] >= min_freq)
                                ]
                
                                # Get conditions from filtered data
                                unique_conditions = sorted(set(
                                    filtered_data['ConditionA'].unique()) |
                                    set(filtered_data['ConditionB'].unique())
                                )
                
                                # Use the same session state as tab 2
                                if 'selected_conditions' not in st.session_state:
                                    st.session_state.selected_conditions = []
                
                                def on_condition_select():
                                    # Update the shared session state
                                    st.session_state.selected_conditions = st.session_state.custom_select
                
                                selected_conditions = st.multiselect(
                                    "Select Initial Conditions",
                                    options=unique_conditions,
                                    default=st.session_state.selected_conditions,
                                    key="custom_select",
                                    on_change=on_condition_select,
                                    help="Choose the starting conditions for trajectory analysis"
                                )
                
                        
                                if selected_conditions:
                                    max_years = math.ceil(filtered_data['MedianDurationYearsWithIQR']
                                                    .apply(lambda x: parse_iqr(x)[0]).max())
                                    time_horizon = st.slider(
                                        "Time Horizon (years)",
                                        min_value=1.0,
                                        max_value=float(max_years),  # Convert to float
                                        value=float(st.session_state.time_horizon),  # Convert to float
                                        step=0.5,  
                                        key="custom_time_horizon",
                                        help="Maximum time period to consider"
                                    )


                                    time_margin = st.slider(
                                        "Time Margin",
                                        0.0, 0.5, st.session_state.time_margin, 0.05,
                                        key="custom_time_margin",
                                        help="Allowable variation in time predictions"
                                    )
                
                                    generate_button = st.button(
                                        "🔄 Generate Network",
                                        key="custom_generate",
                                        help="Click to generate trajectory network"
                                    )
                
                            except Exception as e:
                                st.error(f"Error in custom trajectory analysis: {str(e)}")
                            st.markdown('</div>', unsafe_allow_html=True)
                
                    with main_col:
                        if selected_conditions and generate_button:
                            with st.spinner("🌐 Generating network..."):
                                try:
                                    # Clear previous network
                                    st.session_state.network_html = None
                                    
                                    # Generate new network
                                    html_content = create_network_graph(
                                        filtered_data,
                                        selected_conditions,
                                        min_or,
                                        time_horizon,
                                        time_margin
                                    )
                                    st.session_state.network_html = html_content
                                    st.components.v1.html(html_content, height=800)
                
                                    st.download_button(
                                        label="📥 Download Network",
                                        data=html_content,
                                        file_name="custom_trajectory_network.html",
                                        mime="text/html"
                                    )
                                except Exception as viz_error:
                                    st.error(f"Failed to generate network visualisation: {str(viz_error)}")
                                    st.session_state.network_html = None
                
                        # Display existing network if available
                        elif st.session_state.network_html is not None:
                            st.components.v1.html(st.session_state.network_html, height=800)
                            st.download_button(
                                label="📥 Download Network",
                                data=st.session_state.network_html,
                                file_name="custom_trajectory_network.html",
                                mime="text/html"
                            )

                
            
            
                with tabs[4]:
                    st.header("Cohort Network Analysis")
                    st.markdown("""
                    Visualize relationships between conditions as a network graph. 
                    Node colors represent body systems, and edge thickness indicates association strength.
                    """)
                
                    main_col, control_col = st.columns([3, 1])
                
                    with control_col:
                        with st.container():
                            st.markdown("### Control Panel")
                            
                            # Dynamically set slider ranges based on the loaded data
                            min_or_range = (data['OddsRatio'].min(), data['OddsRatio'].max())
                            min_freq_range = (data['PairFrequency'].min(), data['PairFrequency'].max())
                            
                            # Sliders for filtering
                            min_or = st.slider(
                                "Minimum Odds Ratio",
                                float(min_or_range[0]), float(min_or_range[1]), 2.0, 0.1,
                                key="cohort_network_min_or",
                                help="Filter relationships by minimum odds ratio"
                            )
                
                            min_freq = st.slider(
                                "Minimum Pair Frequency",
                                int(min_freq_range[0]),
                                int(min_freq_range[1]),
                                int(min_freq_range[0]),
                                key="cohort_network_min_freq",
                                help="Minimum number of occurrences required"
                            )
                
                            generate_button = st.button(
                                "🔄 Generate Network",
                                key="cohort_network_generate",
                                help="Create network visualization"
                            )
                
                    with main_col:
                        if generate_button:
                            with st.spinner("🌐 Generating network visualization..."):
                                try:
                                    # Calculate summary statistics
                                    filtered_data = data[
                                        (data['OddsRatio'] >= min_or) &
                                        (data['PairFrequency'] >= min_freq)
                                    ]
                
                                    # Add a check to ensure filtered_data is not empty
                                    if filtered_data.empty:
                                        st.warning("No data matches the current filter criteria. Please adjust the sliders.")
                                        return
                
                                    summary_info = {
                                        'sex': gender,
                                        'age_group': age_group,
                                        'total_patients': total_patients,
                                        'odds_ratio_min': filtered_data['OddsRatio'].min(),
                                        'odds_ratio_max': filtered_data['OddsRatio'].max(),
                                        'min_prevalence': filtered_data['Percentage'].min(),
                                        'min_prevalence_patients': int((filtered_data['Percentage'].min() * total_patients) / 100),
                                        'condition_pairs': len(filtered_data),
                                        'total_odds_ratio': filtered_data['OddsRatio'].sum()
                                    }
                
                                    # Create visualization
                                    html_content = create_network_visualization(data, min_or, min_freq)
                                    
                                    # Display summary information
                                    st.info(f"""
                                    **Filtered Condition Network (min OR: {min_or:.1f}, min frequency: {min_freq:.1f})**
                                    **Sex:** {summary_info['sex']}. 
                                    **Age group:** {summary_info['age_group']}. 
                                    **Total patients with diagnoses in this group:** {summary_info['total_patients']:,}. 
                                    **Odds ratio range:** [{summary_info['odds_ratio_min']:.2f} - {summary_info['odds_ratio_max']:.2f}]. 
                                    **Observed minimum prevalence:** {summary_info['min_prevalence']:.2f}% ({summary_info['min_prevalence_patients']:,} patients). 
                                    **Number of condition pairs shown:** {summary_info['condition_pairs']}. 
                                    **Total sum of odds ratios:** {summary_info['total_odds_ratio']:.2f}.
                                    """)
                                    
                                    # Display network
                                    st.components.v1.html(html_content, height=800)
                                    
                                    # Add download button
                                    st.download_button(
                                        label="📥 Download Network Visualization",
                                        data=html_content,
                                        file_name="condition_network.html",
                                        mime="text/html",
                                        key="cohort_network_download"
                                    )
                                    
                                except Exception as e:
                                    st.error(f"Error generating network visualization: {str(e)}")
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.stop()

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()
        
    add_footer()     

if __name__ == "__main__":
    main()
