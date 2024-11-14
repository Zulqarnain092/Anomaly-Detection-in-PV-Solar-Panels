import streamlit as st
import altair as alt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import re

# Set page configuration
st.set_page_config(
    page_title="Anomaly Detection in Solar Plants",
    page_icon="☀️",
    layout="wide",         # Options: "centered" or "wide"
    initial_sidebar_state="auto"  # Options: "auto", "expanded", "collapsed"
)

# Enhanced Custom CSS for the header and layout adjustments
st.markdown("""
    <style>
        /* General page layout adjustments */
        .reportview-container {
            padding-top: 10px; /* Adjust as needed for top padding */
        }
        .main .block-container {
            max-width: 1400px; /* Set a max width for content */
            padding: 1rem 2rem; /* Control padding for layout */
            margin: auto;
        }

        /* Header container styling */
        .header-container {
            padding: 1.5rem;
            background-color: lightblue;
            color: white;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            text-align: center;
            height: 65px;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        /* Header text styling */
        .header-text {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0;
        }

        /* Anomaly metric container styling */
        .anomaly-metric {
            padding: 1rem;
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            text-align: center;
        }

        /* Metric value text styling */
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #ff4b4b;
        }

        /* Expander header customization */
        .streamlit-expanderHeader {
            font-size: 29px;
        }

        /* Table styling for layout consistency */
        .styled-table {
            font-family: Arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
        }
        .styled-table th {
            background-color: #ADD8E6;
            color: #000000;
            font-weight: bold;
            padding: 8px;
            text-align: center;
        }
        .styled-table td {
            background-color: #D6EAF8;
            padding: 8px;
            text-align: center;
        }
        .styled-table tr:nth-child(even) {
            background-color: #EBF5FB;
        }
        .styled-table tr:hover {
            background-color: #AED6F1;
        }

    </style>
""", unsafe_allow_html=True)


# Function to load and preprocess data for Plant 1
@st.cache_data
def load_data_plant1():
    url = 'https://raw.githubusercontent.com/Zulqarnain092/Anomaly-Detection-in-PV-Solar-Panels/main/solar-power-generation-data/Plant_1_Generation_Data.csv'
    plant1_generation = pd.read_csv(url)

    url2 = 'https://raw.githubusercontent.com/Zulqarnain092/Anomaly-Detection-in-PV-Solar-Panels/main/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv'
    plant1_weather = pd.read_csv(url2)
    
    # Convert dates
    plant1_generation['DATE_TIME'] = pd.to_datetime(plant1_generation['DATE_TIME'], format='%d-%m-%Y %H:%M')
    plant1_weather['DATE_TIME'] = pd.to_datetime(plant1_weather['DATE_TIME'])
    
    # Merge data
    df_solar1 = pd.merge(plant1_generation.drop(columns=['PLANT_ID']),
                        plant1_weather.drop(columns=['PLANT_ID', 'SOURCE_KEY']),
                        on='DATE_TIME')
    
    # Create inverter mapping with explicit count of unique source keys
    source_keys = sorted(df_solar1['SOURCE_KEY'].unique())  # Sort the keys to ensure consistent mapping
    source_key_map = {key: f'inv{str(i+1).zfill(2)}' for i, key in enumerate(source_keys)}
    df_solar1['INVERTER'] = df_solar1['SOURCE_KEY'].map(source_key_map)
    
    # Print number of unique inverters for verification
    print(f"Number of unique inverters in Plant 1: {len(df_solar1['INVERTER'].unique())}")
    
    # Reorder columns
    new_column_order = ['DATE_TIME','INVERTER', 'DC_POWER', 'AC_POWER',
                       'DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE',
                       'MODULE_TEMPERATURE', 'IRRADIATION']
    df_solar1 = df_solar1[new_column_order]
    
    return df_solar1

# Function to load and preprocess data for Plant 2
@st.cache_data
def load_data_plant2():
    url_generation_plant2 = 'https://raw.githubusercontent.com/Zulqarnain092/Anomaly-Detection-in-PV-Solar-Panels/main/solar-power-generation-data/Plant_2_Generation_Data.csv'
    plant2_generation = pd.read_csv(url_generation_plant2)

    url_weather_plant2 = 'https://raw.githubusercontent.com/Zulqarnain092/Anomaly-Detection-in-PV-Solar-Panels/main/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv'
    plant2_weather = pd.read_csv(url_weather_plant2)
    
    df_solar2 = pd.merge(plant2_generation.drop(columns=['PLANT_ID']), 
                        plant2_weather.drop(columns=['PLANT_ID', 'SOURCE_KEY']), 
                        on='DATE_TIME')
    
    # Create inverter mapping with explicit count of unique source keys
    source_keys = sorted(df_solar2['SOURCE_KEY'].unique())  # Sort the keys to ensure consistent mapping
    source_key_map = {key: f'inv{str(i+1).zfill(2)}' for i, key in enumerate(source_keys)}
    df_solar2['INVERTER'] = df_solar2['SOURCE_KEY'].map(source_key_map)
    
    # Print number of unique inverters in Plant 2
    print(f"Number of unique inverters in Plant 2: {len(df_solar2['INVERTER'].unique())}")
    
    new_column_order = ['DATE_TIME','INVERTER', 'DC_POWER', 'AC_POWER',
                       'DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE',
                       'MODULE_TEMPERATURE', 'IRRADIATION']
    df_solar2 = df_solar2[new_column_order]
    
    return df_solar2

@st.cache_resource
def train_isolation_forest(df, plant_num):
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    features = ['AC_POWER', 'IRRADIATION']
    train_features = train[features]
    test_features = test[features]
    
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    if plant_num == 1:
        iso_forest = IsolationForest(contamination=0.01,
                                   max_features=2,
                                   max_samples=0.05,
                                   n_estimators=50,
                                   random_state=42)
    else:
        iso_forest = IsolationForest(contamination=0.03,
                                   max_features=2,
                                   max_samples=0.2,
                                   bootstrap=True,
                                   n_estimators=50,
                                   random_state=8)
    
    iso_forest.fit(train_features_scaled)
    test['predicted_anomaly'] = iso_forest.predict(test_features_scaled)
    test['predicted_anomaly'] = test['predicted_anomaly'].map({1: 0, -1: 1})
    
    if plant_num == 2:
        def post_processing(row):
            if row['AC_POWER'] == 0 and row['IRRADIATION'] > 0:
                return 1
            return row['predicted_anomaly']
        test['predicted_anomaly'] = test.apply(post_processing, axis=1)
    
    return test, scaler

def plot_anomalies_bar_chart(anomalies_by_inverter, selected_inverter):
    # Sort anomalies by count in descending order
    sorted_anomalies = anomalies_by_inverter.sort_values(by='Number of Anomalies', ascending=False)
    
    # Create the bar chart
    fig = px.bar(sorted_anomalies, 
                 x='INVERTER', 
                 y='Number of Anomalies', 
                 title='Number of Anomalies by Inverter')

    # Define colors based on selection
    if selected_inverter == "None":
        # When no inverter is selected, all bars are red with full opacity
        bar_colors = ['rgba(222,45,38,1.0)'] * len(sorted_anomalies)
    else:
        # When an inverter is selected, highlight it and fade others
        bar_colors = ['rgba(222,45,38,1.0)' if inv == selected_inverter 
                     else 'rgba(204,204,204,0.3)'  # Increased transparency for non-selected
                     for inv in sorted_anomalies['INVERTER']]

    # Update the bar colors
    fig.update_traces(marker_color=bar_colors)
    
    # Update layout
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            title="Inverter",
            tickangle=0  # Angle the x-axis labels for better readability
        ),
        yaxis=dict(
            showgrid=False,
            title="Number of Anomalies"
        ),
        # Add more padding at the bottom for angled labels
        margin=dict(b=80),
        # Adjust bar width
        bargap=0.2,
        # Make the plot more prominent
        height=400,
        # Add hover template
        hovermode='closest'
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Anomalies: %{y}<extra></extra>"
    )
    
    return fig

def plot_scatter_ac_vs_irradiation(df, selected_inverter):
    fig = go.Figure()
    
    # Function to determine opacity based on selection
    def get_opacity(inverter):
        if selected_inverter == "None":
            return 1.0  # Full opacity when no inverter is selected
        return 1.0 if inverter == selected_inverter else 0.1
    
    # Plot normal points
    for inverter in df['INVERTER'].unique():
        normal_data = df[(df['INVERTER'] == inverter) & (df['predicted_anomaly'] == 0)]
        # Skip if data is empty
        if not normal_data.empty:
            fig.add_trace(go.Scatter(
                x=normal_data['IRRADIATION'],
                y=normal_data['AC_POWER'],
                mode='markers',
                name=f'{inverter} - Normal' if selected_inverter != "None" else 'Normal',
                marker=dict(
                    color='blue',
                    size=5,
                    opacity=get_opacity(inverter)
                ),
                # Show in legend only for first inverter or selected inverter
                showlegend=(inverter == df['INVERTER'].unique()[0] if selected_inverter == "None" 
                          else inverter == selected_inverter)
            ))
    
    # Plot anomaly points
    for inverter in df['INVERTER'].unique():
        anomaly_data = df[(df['INVERTER'] == inverter) & (df['predicted_anomaly'] == 1)]
        # Skip if data is empty
        if not anomaly_data.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_data['IRRADIATION'],
                y=anomaly_data['AC_POWER'],
                mode='markers',
                name=f'{inverter} - Anomaly' if selected_inverter != "None" else 'Anomaly',
                marker=dict(
                    color='red',
                    size=7,
                    opacity=get_opacity(inverter)
                ),
                # Show in legend only for first inverter or selected inverter
                showlegend=(inverter == df['INVERTER'].unique()[0] if selected_inverter == "None" 
                          else inverter == selected_inverter)
            ))
    
    # Update layout
    fig.update_layout(
        title="AC Power vs Irradiation",
        xaxis_title="Irradiation",
        yaxis_title="AC Power (kW)",
        height=650,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        # Position the legend outside the chart area on the right
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,  # Position just outside the chart area
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    return fig

def plot_irradiation_and_ac_power(df, inverter, test_data):
    # Filter data for the selected inverter
    df_filtered = df[df['INVERTER'] == inverter]
    
    # Get anomaly points from test_data
    anomaly_points = test_data[
        (test_data['INVERTER'] == inverter) & 
        (test_data['predicted_anomaly'] == 1)
    ]

    fig = go.Figure()
    
    # Add original traces
    fig.add_traces([
        go.Scatter(
            x=df_filtered['DATE_TIME'], 
            y=df_filtered['IRRADIATION'],
            mode='lines', 
            name='Irradiation', 
            yaxis='y1'
        ),
        go.Scatter(
            x=df_filtered['DATE_TIME'], 
            y=df_filtered['AC_POWER'],
            mode='lines', 
            name='AC Power (kW)', 
            yaxis='y2'
        )
    ])
    
    # Add anomaly points
    fig.add_traces([
        go.Scatter(
            x=anomaly_points['DATE_TIME'],
            y=anomaly_points['IRRADIATION'],
            mode='markers',
            name='Irradiation Anomalies',
            marker=dict(color='red', size=8, symbol='x'),
            yaxis='y1'
        ),
        go.Scatter(
            x=anomaly_points['DATE_TIME'],
            y=anomaly_points['AC_POWER'],
            mode='markers',
            name='AC Power Anomalies',
            marker=dict(color='red', size=8, symbol='star'),
            yaxis='y2'
        )
    ])
    
    fig.update_layout(
        title=f'Irradiation and AC Power for Inverter: {inverter}',
        xaxis_title='Date',
        yaxis=dict(title='Irradiation', showgrid=False),
        yaxis2=dict(title='AC Power (kW)', overlaying='y', side='right', showgrid=False),
        height=350,
        showlegend=True
    )
    
    return fig


# Injecting custom CSS for rounded rectangles
st.markdown("""
    <style>
    .rounded-container {
        padding: 15px;
        background-color: #f9f9f9;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .rounded-column {
        padding: 15px;
        background-color: #f2f2f2;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

def display_overview():
    st.markdown("## Anomaly Detection in Photovoltaic Solar Panel with Isolation Forest")

    # Container with rounded edges
    background_container = st.container()
    with background_container:
        

        #st.markdown("#### Solar Energy Demand in India over Time")
        st.markdown("""""")
        

        # Creating the dataset
        data = {
            'Year': ['2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25'],
            'Solar': [0.04, 0.94, 1.69, 2.63, 3.74, 6.76, 12.29, 0, 0, 0, 0, 0, 0, 0, 0],
            'Solar - Ground Mounted': [0, 0, 0, 0, 0, 0, 0, 20.59, 26.38, 32.11, 35.65, 45.79, 55.51, 66.99, 72.68],
            'Solar - Offgrid': [0, 0, 0, 0, 0, 0, 0, 0.69, 0.92, 0.98, 1.15, 1.56, 2.39, 2.96, 3.78],
            'Solar - Rooftop': [0, 0, 0, 0, 0, 0, 0, 1.06, 1.8, 2.52, 4.44, 6.65, 8.88, 11.87, 14.3]
        }

        # Creating a DataFrame
        df = pd.DataFrame(data)

        # Melting the DataFrame to have a long format
        df_melted = df.melt('Year', var_name='Category', value_name='Value')

        # Plotting the bar plot using Altair
        chart = alt.Chart(df_melted).mark_bar().encode(
            x=alt.X('Year:N', title='Year', axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('Value:Q', title='Installed Capacity (in GW)'),
            color='Category:N',
            tooltip=['Year', 'Category', 'Value']
        ).properties(
            title=alt.TitleParams(text='Growth of Installed Capacity (GW) from 2010-11 to 2024-25', align='center', anchor='middle')
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            grid=False
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
        
        st.markdown(
            """
            <p style='text-align: center;'>
                Source: <a href='https://iced.niti.gov.in' target='_blank'>India's Climate and Energy Dashboard</a>
            </p>
            """, 
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Columns with rounded edges
    col1, col2 = st.columns(2)

    with col1:
        
        st.markdown("### Issues with PV Solar Panel")
        st.markdown("##### **Core Issue:** Difficulty detecting energy loss in PV solar panels due to unconditional weather raises maintenance costs and limits performance.")
        st.markdown("**Breakdown of the Problem:**")
        st.markdown("- **Energy Loss and Maintenance:** Energy losses in PV systems not only reduce output but also increase operational and maintenance costs, particularly for large installations where identifying issues manually is slow and labor-intensive.")
        st.markdown("- **Delayed Anomaly Detection:** Relying on periodic inspections means that issues often go unnoticed until they become severe, leading to reactive instead of proactive maintenance.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        
        st.markdown("### Project Objective")
        st.markdown("1. **To develop an anomaly detection model that accurately identifies irregularities in solar power generation.**")
        st.markdown("2. **To provide actionable insights and recommendations for maintenance and operational improvements based on the anomaly detection results.**")
        st.markdown("</div>", unsafe_allow_html=True)

    # Dataset Information container
    dataset_container = st.container()
    with dataset_container:
        st.markdown("### Dataset Information")
        
        st.markdown("""
            The data utilizes a publicly available dataset owned by <a href='https://www.kaggle.com/datasets/anikannal/solar-power-generation-data/data' target='_blank'><i>Kannal (2017)</i></a> 
                    which spans a 30-day period from May 15, 2020, to June 17, 2020, collected from two solar power plants in India. 
                    This dataset consists of two main components: power generation data and sensor readings.

            **Power Generation Data**: Collected from 22 individual inverters connected to multiple lines of solar panels, recorded every 15 minutes.
            
            **Sensor Readings**: Gathered at the plant level using strategically placed sensors, also recorded every 15 minutes.
            ### Key Features:
            - **DATE_TIME**: The timestamp for each observation, captured at 15-minute intervals.
            - **PLANT_ID**: A unique identifier assigned to each plant, remaining consistent throughout the dataset.
            - **SOURCE_KEY**: The inverter's identifier, with each inverter managing several lines of solar panels.
            - **DC_POWER**: Direct current (DC) power output in kilowatts (kW), logged every 15 minutes.
            - **AC_POWER**: Alternating current (AC) power output in kilowatts (kW), also captured at 15-minute intervals.
            - **TOTAL_YIELD**: The cumulative energy yield (in kWh) of each inverter up to the recorded time.
            - **AMBIENT_TEMPERATURE**: The surrounding air temperature at the plant.
            - **MODULE_TEMPERATURE**: The temperature measured directly from a solar module attached to the sensor array.
            - **IRRADIATION**: Intensity of sunlight reaching the panels during the 15-minute interval.
            """, unsafe_allow_html=True)


    # Seasons container
    location_container = st.container()
    with location_container:
        
        st.markdown("### Seasons in India")
        st.markdown("""
            **Seasons in India**:
            - **Winter** (Dec-Apr)
            - **Summer/Pre-Monsoon** (Apr-Jun)
            - **Monsoon** (Jun-Sep)
            - **Post-Monsoon** (Oct-Dec)
            India's climate is largely influenced by the **monsoon regime**, resulting in a distinctive **dry and rainy season**.
        """)
        st.image(
            "https://www.climatestotravel.com/Images/india/webp/Climates-India.webp",
            use_column_width=False
        )
        st.markdown("""
            _(Source: [Climates to Travel](https://www.climatestotravel.com/climate/india))_
                    
            **Rainfall patterns vary greatly across regions:**
            - The **south-west coast** (e.g., Mumbai, Goa, Mangalore) receives over **2,000 mm** (80 in) annually.
            - The **north-east region** (e.g., Kolkata and areas near Bangladesh) sees more than **1,500 mm** (60 in) of rain per year.
            - At the **Himalayan foothills**, some areas are among the rainiest in the world.
            - Conversely, **northwestern India** is quite arid, with western Rajasthan receiving less than **300 mm** (12 in) of rain annually.
            - **Central and southern inland areas** (e.g., Hyderabad, Bangalore) typically receive between **800-1,000 mm** (32-40 in) annually.

            The **hottest period** generally occurs **April to mid-June**, just before the monsoon, with temperatures often reaching **45 °C (113 °F)** in inland regions. **Summer in India** (May to August) is marked by **high temperatures**, typically between **37.8°C (100°F)** and **46.1°C (115°F)**. 
                    
            Interior regions like **Rajasthan** experience extreme heat, with record highs, such as **51°C (123.8°F)** recorded in Phalodi, Rajasthan, in May 2019. The southwest monsoon arriving in June brings relief, with coastal areas benefiting from cooling sea breezes, although **desert areas** remain dry with less than **254 mm** (10 in) of rain annually.

            _(Source: [Weather Atlas](https://www.weather-atlas.com/en/india-climate))_
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)
   
    iforest_container = st.container()

    with iforest_container:
        st.markdown("### Isolation Forest")

        html_string = '''
            <div style="font-size: x-large;">
                <p> According to <a href="https://doi.org/10.1109/ICIC3S61846.2024.10602838"><i>Ravinder et al. (2024)</i></a>, Isolation Forest is an unsupervised learning approach that focuses on detecting anomalies rather than normal data points.
                 It operates on the premise that anomalies are "few and different," making them easier to isolate.</p>
                <p><b>Step 1:</b> Construct multiple binary trees, each from random data subsets.</p>
                <p><b>Step 2:</b> At each node in the tree, randomly select a feature and choose a split value between the minimum and maximum values of that feature.</p>
                <p><b>Step 3:</b> For each data point, calculate the path length from the terminal node to the root node.</p>
            </div>
        '''

        st.markdown(html_string, unsafe_allow_html=True)


        html_string = '''
            <div style="text-align: center;">
                <img src="https://www.nomidl.com/wp-content/uploads/2022/11/image-4.png" style="max-width:50%;">
                <p style="text-align: center; margin-top: 10px;">Isolation Forest Algorithm. Source: <a href="https://www.nomidl.com/machine-learning/outlier-detection-methods-in-machine-learning/"><i>Naveen (2023)</i></a>.</p>
            </div>
        '''

        st.markdown(html_string, unsafe_allow_html=True)

        st.markdown("""### Advantages of Isolation Forest for Anomaly Detection""")

        st.write("""
            According to <a href="https://doi.org/10.3390/en15031082"><i>Ibrahim et al. (2022)</i></a>, the Isolation Forest algorithm offers several advantages for anomaly detection, making it an attractive choice for various applications. 
            These benefits include:
        """, unsafe_allow_html=True)

        st.markdown("- **Unsupervised Learning:** It does not require labelled data for anomaly detection, which is particularly useful when obtaining labelled data is difficult or expensive.")
        st.markdown("- **Efficiency:** The algorithm is computationally fast due to its tree structure, which is built by randomly selecting features, making it highly suitable for large datasets.")
        st.markdown("- **Robustness to Noise:** It is capable of detecting anomalies even in the presence of noise, as it focuses on the structure of the data rather than its distribution.")
        st.markdown("- **Parameter Flexibility:** It offers flexibility by allowing the tuning of parameters such as the number of trees (estimators) and the contamination rate (the expected proportion of outliers), which helps in adapting the model to specific dataset characteristics.")


    # Related Work and Summary Table Container
    related_work_container = st.container()

    with related_work_container:
        st.markdown("### Related Works on Anomaly Detection")

        data_1 = {
            "Author": [
                '<a href="https://doi.org/10.1109/icrest57604.2023.10070033"><i>Kabir et al. (2023)</i></a>',
                '<a href="https://doi.org/10.1109/cist56084.2023.10409931"><i>Hairach et al. (2023)</i></a>',
                '<a href="https://doi.org/10.3390/en15031082"><i>Ibrahim et al. (2023)</i></a>',
                '<a href="https://doi.org/10.1109/ICIC3S61846.2024.10602838"><i>Ravinder et al. (2024)</i></a>',
                '<a href="https://doi.org/10.1109/iccpct61902.2024.10673338"><i>Sharma and Grover (2024)</i></a>'
            ],
            "Algorithm Used": ["Isolation Forest", "Isolation Forest, Local Outlier Factor (LOF), K-Means, and DBSCAN", 
                            "AutoEncoder Long Short-Term Memory (AE-LSTM), Facebook-Prophet, and Isolation Forest", 
                            "Isolation Forest, One-Class SVM", "Isolation Forest, AutoEncoder"],
            "Application": ["Solar PV system", "Solar PV system", "Solar PV system", "Power Consumption", "Cybersecurity enhancement through network traffic anomalies"],
            "Remarks": ["Achieved accuracy of 0.9886", "K-Means and DBSCAN outperformed other models when properly tuned.",
                        "AE-LSTM achieved the highest accuracy (0.8963). However, isolation forest model can be improved.",
                        "Isolation Forest outperformed One-Class SVM, achieving 98.99% accuracy.",
                        "Isolation Forest showed higher Anomaly Detection Rate (ADR) of 85% and a Detection Consistency (DC) of 90% than AutoEncoder."]
        }

        df1 = pd.DataFrame(data_1)

        # Custom CSS for Table Styling
        st.markdown("""
            <style>
            .styled-table {
                font-family: Arial, sans-serif;
                border-collapse: collapse;
                width: 100%;
            }
            .styled-table th {
                background-color: #ADD8E6;
                color: #000000;
                font-weight: bold;
                padding: 8px;
                text-align:center;
            }
            .styled-table td {
                background-color: #D6EAF8;
                padding: 8px;
                text-align:center;
            }
            .styled-table tr:nth-child(even) {
                background-color: #EBF5FB;
            }
            .styled-table tr:hover {
                background-color: #AED6F1;
            }
            </style>
        """, unsafe_allow_html=True)

        # Displaying Table 1 with Custom CSS and Hyperlinks
        st.markdown(df1.to_html(escape=False, index=False, classes="styled-table"), unsafe_allow_html=True)
        st.markdown("""""")

        st.markdown("### Summary of Works by Authors")

        data_2 = {
            "Authors \\ Algorithms": ["<b>Isolation Forest</b>", "Local Outlier Factor (LOF)", "K-Means", "DBSCAN", "AutoEncoder Long Short-Term Memory (AE-LSTM)", "AutoEncoder", "Facebook-Prophet", "One-Class SVM"],
            '<a href="https://doi.org/10.1109/icrest57604.2023.10070033"><i>Kabir et al. (2023)</i></a>': ["✔", "", "", "", "", "", "", ""],
            '<a href="https://doi.org/10.1109/cist56084.2023.10409931"><i>Hairach et al. (2023)</i></a>': ["✔", "✔", "✔", "✔", "", "", "", ""],
            '<a href="https://doi.org/10.3390/en15031082"><i>Ibrahim et al. (2023)</i></a>': ["✔", "", "", "", "✔", "", "✔", ""],
            '<a href="https://doi.org/10.1109/ICIC3S61846.2024.10602838"><i>Ravinder et al. (2024)</i></a>': ["✔", "", "", "", "", "", "", "✔"],
            '<a href="https://doi.org/10.1109/iccpct61902.2024.10673338"><i>Sharma and Grover (2024)</i></a>': ["✔", "", "", "", "", "✔", "", ""],
            "Total": ['<b>5</b>', 1, 1, 1, 1, 1, 1, 1]
        }

        df2 = pd.DataFrame(data_2)

        # Displaying Table 2 with Custom CSS and Hyperlinks
        st.markdown(df2.to_html(escape=False, index=False, classes="styled-table"), unsafe_allow_html=True)
        st.markdown("""""")
        st.markdown("""""")



def display_plant1():

    st.title("Plant 1 Anomaly Report")

    # load data and train the model
    df_solar1 = load_data_plant1()
    test_data, scaler = train_isolation_forest(df_solar1, 1)

    # Inverter selection with anomaly count
    sorted_inverters = sorted(test_data['INVERTER'].unique(), 
                            key=lambda x: int(re.findall(r'\d+', x)[0]))
    selected_inverter = st.selectbox("Select Inverter", ["None"] + sorted_inverters)

    # Calculate and display anomalies
    anomalies_by_inverter = test_data[test_data['predicted_anomaly'] == 1].groupby('INVERTER').size().reset_index(name='Number of Anomalies')
    total_anomalies = anomalies_by_inverter['Number of Anomalies'].sum()

    # Show anomaly count for selected inverter
    if selected_inverter != "None":
        inverter_anomalies = anomalies_by_inverter[anomalies_by_inverter['INVERTER'] == selected_inverter]['Number of Anomalies'].iloc[0]
        st.markdown(f"""
            <div class="anomaly-metric">
                <div>Anomalies detected for {selected_inverter}</div>
                <div class="metric-value">{inverter_anomalies}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Show total anomalies when None is selected
        st.markdown(f"""
            <div class="anomaly-metric">
                <div>Total anomalies detected</div>
                <div class="metric-value">{total_anomalies}</div>
            </div>
        """, unsafe_allow_html=True)

    
    # Display anomalies 
    st.plotly_chart(plot_scatter_ac_vs_irradiation(test_data, selected_inverter),
                   use_container_width=True)
    
    st.markdown("""
        **Insights:** Sunlight intensity (irradiation) shows a **strong positive correlation** with AC power. 
        This indicates that more sunlight received by the solar panel results in greater electricity generation.
    """)

    st.markdown("### List of Anomalies")
    if selected_inverter == "None":
        anomalies_df = test_data[test_data['predicted_anomaly'] == 1]
    else:
        anomalies_df = test_data[(test_data['INVERTER'] == selected_inverter) & 
                               (test_data['predicted_anomaly'] == 1)]
    st.dataframe(anomalies_df)
    
    # Plots
    st.plotly_chart(plot_anomalies_bar_chart(anomalies_by_inverter, selected_inverter),
                   use_container_width=True)
    
    
    
    if selected_inverter != "None":
        st.plotly_chart(plot_irradiation_and_ac_power(df_solar1, selected_inverter,test_data),
                       use_container_width=True)

def display_plant2():
    
    # Code for Plant 2 analysis
    st.title("Plant 2 Anomaly Report")

    df_solar2 = load_data_plant2()
    test_data, scaler = train_isolation_forest(df_solar2, 2)

    # Inverter selection with anomaly count
    sorted_inverters = sorted(test_data['INVERTER'].unique(), 
                            key=lambda x: int(re.findall(r'\d+', x)[0]))
    selected_inverter = st.selectbox("Select Inverter", ["None"] + sorted_inverters)

    # Calculate and display anomalies
    anomalies_by_inverter = test_data[test_data['predicted_anomaly'] == 1].groupby('INVERTER').size().reset_index(name='Number of Anomalies')
    total_anomalies = anomalies_by_inverter['Number of Anomalies'].sum()

    # Show anomaly count for selected inverter
    if selected_inverter != "None":
        inverter_anomalies = anomalies_by_inverter[anomalies_by_inverter['INVERTER'] == selected_inverter]['Number of Anomalies'].iloc[0]
        st.markdown(f"""
            <div class="anomaly-metric">
                <div>Anomalies detected for {selected_inverter}</div>
                <div class="metric-value">{inverter_anomalies}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Show total anomalies when None is selected
        st.markdown(f"""
            <div class="anomaly-metric">
                <div>Total anomalies detected</div>
                <div class="metric-value">{total_anomalies}</div>
            </div>
        """, unsafe_allow_html=True)

    
    # Display anomalies dataframe
    st.plotly_chart(plot_scatter_ac_vs_irradiation(test_data, selected_inverter),
                   use_container_width=True)
    
    st.markdown("""
        **Insights:** Sunlight intensity (irradiation) shows a **strong positive correlation** with AC power. 
        This indicates that more sunlight received by the solar panel results in greater electricity generation.
    """)

    st.markdown("### List of Anomalies")
    if selected_inverter == "None":
        anomalies_df = test_data[test_data['predicted_anomaly'] == 1]
    else:
        anomalies_df = test_data[(test_data['INVERTER'] == selected_inverter) & 
                               (test_data['predicted_anomaly'] == 1)]
    st.dataframe(anomalies_df)
    
    # Plots
    st.plotly_chart(plot_anomalies_bar_chart(anomalies_by_inverter, selected_inverter),
                   use_container_width=True)
    
    
    
    if selected_inverter != "None":
        st.plotly_chart(plot_irradiation_and_ac_power(df_solar2, selected_inverter,test_data),
                       use_container_width=True)

def display_conclusion():

    # Create a container for the findings
    findings_container = st.container()

    with findings_container:
        st.header("Findings of Anomalies in PV Solar Panels")

        # Section 1: Analysis of Anomaly Detection Patterns (Comparing Plant 1 and Plant 2)
        st.subheader("1. Analysis of Anomaly Detection Patterns")

        data_1 = {
            "Aspects/Plants": ["<b>Anomaly Density</b>", "<b>Distribution Patterns</b>", "<b>Low Output Conditions</b>"],
            "Plant 1": [
                "Anomalies are less frequent",
                "Anomalies stay close to expected power-irradiation values",
                "Few anomalies with low power output"
            ],
            "Plant 2": [
                "Higher density of anomalies across a broader range of irradiation levels",
                "Many anomalies are far from the expected trend, especially at lower power levels",
                "Many anomalies with low power output despite sufficient irradiation"
            ]
        }

        df1 = pd.DataFrame(data_1)

        # Custom CSS for Table Styling
        st.markdown("""
            <style>
            .styled-table {
                font-family: Arial, sans-serif;
                border-collapse: collapse;
                width: 100%;
            }
            .styled-table th {
                background-color: #ADD8E6;
                color: #000000;
                font-weight: bold;
                padding: 8px;
                text-align:center;
            }
            .styled-table td {
                background-color: #D6EAF8;
                padding: 8px;
                text-align:center;
            }
            .styled-table tr:nth-child(even) {
                background-color: #EBF5FB;
            }
            .styled-table tr:hover {
                background-color: #AED6F1;
            }
            </style>
        """, unsafe_allow_html=True)

        # Displaying Table 1 with Custom CSS
        st.markdown(df1.to_html(escape=False, index=False, classes="styled-table"), unsafe_allow_html=True)
        st.markdown("""""")

        # Section 2: Distribution of Anomalies by Inverter (Comparing Anomaly Counts and Distribution)
        st.subheader("2. Distribution of Anomalies by Inverter")

        data_2 = {
            "Aspects/Plants": ["<b>Anomaly Counts</b>", "<b>Anomaly Distribution</b>"],
            "Plant 1": [
                "Most inverters showing <b>fewer than 10 anomalies</b>",
                "Most inverters have a similar amount of anomalies"
            ],
            "Plant 2": [
                "Many inverters show high anomaly counts of <b>over 70</b>",
                "Anomalies are widespread, affecting nearly all inverters"
            ]
        }

        df2 = pd.DataFrame(data_2)

        # Displaying Table 2 with Custom CSS
        st.markdown(df2.to_html(escape=False, index=False, classes="styled-table"), unsafe_allow_html=True)
        st.markdown("""""")
        st.markdown("""""")

    # Create a container for the findings
    mitigation_container = st.container()

    with mitigation_container:
        st.header("Mitigation Strategies for Identified Anomalies")

        st.markdown("""
        Based on the anomaly analysis across both plants, several mitigation actions are recommended to address the operational issues detected in Plant 1 and Plant 2:
        
        ### 1. Shading Assessment
        - **Objective**: Identify if any objects, such as trees or nearby structures, cast shadows on the solar panels, which could obstruct sunlight and lead to inconsistent power generation.
        - **Action**: Conduct a site survey to assess shading patterns throughout the day. Adjust or remove obstructions where possible to ensure uninterrupted sunlight exposure.

        ### 2. Inspect Panel Age and Condition
        - **Objective**: Determine if the efficiency of the solar panels has decreased due to aging or physical wear.
        - **Action**: Examine panels for signs of wear, degradation, or damage. Replace or repair older panels to maintain consistent power output and reduce anomaly occurrences.

        ### 3. Address Inverter Issues
        - **Objective**: Ensure that inverters are functioning optimally, as they play a critical role in converting DC to AC power.
        - **Actions**:
            - **Check for Overheating**: Inverters are prone to overheating, which can reduce efficiency. Ensure proper ventilation and consider installing cooling mechanisms if needed.
            - **Electrical Faults**: Regularly inspect for any wiring issues or loose connections that could affect inverter performance.
            - **Software and Firmware Updates**: Keep inverter software updated to prevent glitches that may result in misclassification of power output data.

        ### 4. Optimize Panel Orientation and Tilt
        - **Objective**: Ensure that panels are optimally positioned to capture maximum sunlight.
        - **Action**: Review the positioning of solar panels, adjusting tilt and orientation to maximize solar irradiance based on geographic location and seasonal sunlight patterns.

        ### 5. Regular Maintenance and Calibration
        - **Objective**: Minimize the occurrence of false positives and improve the accuracy of anomaly detection by ensuring equipment is regularly maintained and calibrated.
        - **Actions**: Schedule periodic maintenance for all system components, including sensors and inverters, to detect and address potential issues early.
        """)


def main():
    pages = {
        "Overview": display_overview,
        "Plant 1": display_plant1,
        "Plant 2": display_plant2,
        "Conclusion": display_conclusion
    }

    st.sidebar.title("Solar Plant Analysis")
    selection = st.sidebar.selectbox("Navigate to", list(pages.keys()))

    pages[selection]()

if __name__ == "__main__":
    main()
