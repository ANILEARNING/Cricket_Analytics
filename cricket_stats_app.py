import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from fpdf import FPDF
from PIL import Image

# Load Match List
def load_match_list():
    with open("match_list.json", "r") as file:
        return json.load(file)

# Fetch Data
def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None

# Fetch & Store Data for Selected Match
def fetch_match_data(match_id, base_url):
    paths = {"commentary": "innings%2F1%2Fcommentary", "wagon": "wagons", "statistics": "statistics"}
    all_data = {}
    for key, path in paths.items():
        url = base_url.replace("{MatchId}", match_id).replace("{Path}", path)
        data = fetch_data(url)
        if data:
            all_data[key] = data
    return all_data

# Create DataFrames
def create_dataframes(all_data):
    # Safely extract commentary data
    commentary_data = all_data.get("commentary", {}).get("response", [])

    if isinstance(commentary_data, list):
        try:
            df_commentary = pd.DataFrame(commentary_data)
        except ValueError:
            # Handle missing or irregular data
            df_commentary = pd.DataFrame.from_dict(commentary_data, orient="index").reset_index()
    else:
        df_commentary = pd.DataFrame()  # Empty DataFrame if data is not valid

    # Safely extract wagon data
    wagon_data = all_data.get("wagon", {}).get("response", {}).get("innings", [])
    df_wagon = pd.DataFrame(wagon_data) if isinstance(wagon_data, list) else pd.DataFrame()

    # Safely extract statistics data
    statistics_data = all_data.get("statistics", {}).get("response", {}).get("innings", [])
    df_statistics = pd.DataFrame(statistics_data) if isinstance(statistics_data, list) else pd.DataFrame()

    return df_commentary, df_wagon, df_statistics

# Function to save figure to memory and add to PDF
def save_fig_to_pdf(fig, pdf, x, y, w=180, h=100):
    # Save figure to a temporary file
    temp_file = "temp_plot.png"
    fig.savefig(temp_file, format='png', bbox_inches="tight", dpi=300)
    
    # Add image to PDF
    pdf.image(temp_file, x=x, y=y, w=w, h=h)
    
    # Close the figure to free up memory
    plt.close(fig)

# Manhattan Chart (Runs per Over)
def plot_manhattan_chart(df_statistics, pdf=None):
    if df_statistics.empty:
        st.warning("No statistics available.")
        return
    
    try:
        df_manhattan = pd.DataFrame(df_statistics.iloc[0]["statistics"]["manhattan"])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x="over", y="runs", data=df_manhattan, color="blue", ax=ax)
        ax.set_title("Manhattan Chart - Runs per Over", fontsize=15)
        ax.set_xlabel("Over", fontsize=12)
        ax.set_ylabel("Runs", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)

        if pdf:
            save_fig_to_pdf(fig, pdf, x=10, y=50)
    except Exception as e:
        st.error(f"Error creating Manhattan chart: {e}")

# Worm Chart (Total Runs over Time)
def plot_worm_chart(df_statistics, pdf=None):
    if df_statistics.empty:
        st.warning("No statistics available.")
        return
    
    try:
        df_worm = pd.DataFrame(df_statistics.iloc[0]["statistics"]["worm"])

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x="over", y="runs", data=df_worm, marker="o", color="green", ax=ax)
        ax.set_title("Worm Chart - Runs Progression", fontsize=15)
        ax.set_xlabel("Over", fontsize=12)
        ax.set_ylabel("Cumulative Runs", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)

        if pdf:
            save_fig_to_pdf(fig, pdf, x=10, y=100)
    except Exception as e:
        st.error(f"Error creating Worm chart: {e}")

# Boundary Analysis
def plot_boundary_chart(df_statistics, pdf=None):
    if df_statistics.empty:
        st.warning("No statistics available.")
        return
    
    try:
        df_boundary = pd.DataFrame(df_statistics.iloc[0]["statistics"]["runtypes"])
        df_boundary = df_boundary[df_boundary["key"].isin(["run4", "run6"])]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x="name", y="value", data=df_boundary, palette=["lightblue", "orange"], ax=ax)
        ax.set_title("Boundary Breakdown (Fours & Sixes)", fontsize=15)
        ax.set_xlabel("Boundary Type", fontsize=12)
        ax.set_ylabel("Number of Boundaries", fontsize=12)
        plt.tight_layout()
        
        st.pyplot(fig)

        if pdf:
            save_fig_to_pdf(fig, pdf, x=10, y=150)
    except Exception as e:
        st.error(f"Error creating Boundary chart: {e}")

# Bowler Economy
def plot_bowler_economy_chart(df_statistics, pdf=None):
    if df_statistics.empty:
        st.warning("No statistics available.")
        return

    try:
        # Check if 'bowlers' key exists
        if "bowlers" not in df_statistics.iloc[0]["statistics"]:
            st.warning("No bowler data available.")
            return
        
        df_bowlers = pd.DataFrame(df_statistics.iloc[0]["statistics"]["bowlers"])

        if df_bowlers.empty:
            st.warning("No bowler data available.")
            return

        # Plot Bowler Economy
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x="name", y="econ", data=df_bowlers.sort_values("econ"), palette="coolwarm", ax=ax)
        ax.set_title("Bowler Economy Rate", fontsize=15)
        ax.set_xlabel("Bowler", fontsize=12)
        ax.set_ylabel("Economy Rate", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)

        if pdf:
            save_fig_to_pdf(fig, pdf, x=10, y=200)
    except Exception as e:
        st.error(f"Error creating Bowler Economy chart: {e}")

# Generate PDF Report with Charts & KPIs
def generate_pdf(df_statistics):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "Cricket Match Report", ln=True, align="C")
        pdf.ln(10)

        # KPIs
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Key Performance Indicators (KPIs)", ln=True)
        pdf.set_font("Arial", size=10)
        innings = df_statistics.iloc[0]
        
        # Safely extract KPIs with error handling
        try:
            total_runs = innings.get('runs', 'N/A')
            total_wickets = innings.get('wickets', 'N/A')
            total_overs = innings.get('overs', 'N/A')
            
            run_rates = innings.get('statistics', {}).get('runrates', [])
            run_rate = run_rates[-1]['runrate'] if run_rates else 'N/A'
            
            run_types = innings.get('statistics', {}).get('runtypes', [])
            fours = sum(x['value'] for x in run_types if x['key'] == 'run4')
            sixes = sum(x['value'] for x in run_types if x['key'] == 'run6')
        except Exception as e:
            st.error(f"Error extracting KPIs: {e}")
            total_runs = total_wickets = total_overs = run_rate = fours = sixes = 'N/A'

        pdf.cell(200, 10, f"Total Runs: {total_runs}", ln=True)
        pdf.cell(200, 10, f"Total Wickets: {total_wickets}", ln=True)
        pdf.cell(200, 10, f"Overs Played: {total_overs}", ln=True)
        pdf.cell(200, 10, f"Run Rate: {run_rate}", ln=True)
        pdf.cell(200, 10, f"Fours: {fours}", ln=True)
        pdf.cell(200, 10, f"Sixes: {sixes}", ln=True)
        
        # Charts
        plot_manhattan_chart(df_statistics, pdf)
        plot_worm_chart(df_statistics, pdf)
        plot_boundary_chart(df_statistics, pdf)
        plot_bowler_economy_chart(df_statistics, pdf)

        # Save PDF
        pdf_filename = "match_report.pdf"
        pdf.output(pdf_filename)
        return pdf_filename
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# Streamlit UI
def main():
    st.set_page_config(page_title="🏏 Cricket Analytics", layout="wide", page_icon="🏏")

    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        color: #262730;
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏏 Cricket Match Analytics Dashboard")
    st.markdown("---")

    # Load Matches
    matches = load_match_list()
    match_id = st.selectbox("Select Match", list(matches.keys()), format_func=lambda x: matches[x])

    if match_id:
        # Add a spinner during data loading
        with st.spinner('Fetching match data...'):
            base_url = st.secrets["api_url"]
            all_data = fetch_match_data(match_id, base_url)
        
        if all_data:
            df_commentary, df_wagon, df_statistics = create_dataframes(all_data)

            # KPI Metrics with improved styling
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Runs", df_statistics.iloc[0]["runs"])
            with col2:
                st.metric("Wickets", df_statistics.iloc[0]["wickets"])
            with col3:
                st.metric("Overs", df_statistics.iloc[0]["overs"])
            with col4:
                st.metric("Run Rate", df_statistics.iloc[0]["statistics"]["runrates"][-1]["runrate"])

            st.markdown("---")

            # Generate Visualizations
            st.subheader("📊 Match Analysis")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs([
                "Manhattan Chart", 
                "Worm Chart", 
                "Boundary Analysis", 
                "Bowler Economy"
            ])

            with tab1:
                plot_manhattan_chart(df_statistics, None)
            
            with tab2:
                plot_worm_chart(df_statistics, None)
            
            with tab3:
                plot_boundary_chart(df_statistics, None)
            
            with tab4:
                plot_bowler_economy_chart(df_statistics, None)

            # Generate Report Button
            st.markdown("---")
            if st.button("📄 Generate PDF Report"):
                pdf_file = generate_pdf(df_statistics)
                if pdf_file:
                    with open(pdf_file, "rb") as file:
                        st.download_button(
                            "Download Detailed Match Report", 
                            file, 
                            "match_report.pdf", 
                            mime="application/pdf",
                            help="A comprehensive PDF report with match statistics and charts"
                        )
                else:
                    st.error("Failed to generate PDF report. Please try again.")
        else:
            st.error("Unable to fetch match data. Please check your connection or try another match.")

if __name__ == "__main__":
    main()
# import streamlit as st
# import requests
# import json
# import pandas as pd
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# from fpdf import FPDF

# # Load Match List
# def load_match_list():
#     with open("match_list.json", "r") as file:
#         return json.load(file)

# # Fetch Data
# def fetch_data(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error fetching data: {e}")
#         return None

# # Fetch & Store Data for Selected Match
# def fetch_match_data(match_id, base_url):
#     paths = {
#         "commentary": "innings%2F1%2Fcommentary",
#         "wagon": "wagons",
#         "statistics": "statistics"
#     }
#     all_data = {}
#     for key, path in paths.items():
#         url = base_url.replace("{MatchId}", match_id).replace("{Path}", path)
#         data = fetch_data(url)
#         if data:
#             all_data[key] = data
#     return all_data

# # Create DataFrames
# def create_dataframes(all_data):
#     # Safely extract commentary data
#     commentary_data = all_data.get("commentary", {}).get("response", [])
#     df_commentary = pd.DataFrame(commentary_data) if isinstance(commentary_data, list) else pd.DataFrame()

#     # Safely extract wagon data
#     wagon_data = all_data.get("wagon", {}).get("response", {}).get("innings", [])
#     df_wagon = pd.DataFrame(wagon_data) if isinstance(wagon_data, list) else pd.DataFrame()

#     # Safely extract statistics data
#     statistics_data = all_data.get("statistics", {}).get("response", {}).get("innings", [])
#     df_statistics = pd.DataFrame(statistics_data) if isinstance(statistics_data, list) else pd.DataFrame()

#     return df_commentary, df_wagon, df_statistics


# # Manhattan Chart (Runs per Over)
# def plot_manhattan_chart(df_statistics):
#     if df_statistics.empty:
#         st.warning("No statistics available.")
#         return
#     df_manhattan = pd.DataFrame(df_statistics.iloc[0]["statistics"]["manhattan"])
#     plt.figure(figsize=(8, 4))
#     sns.barplot(x="over", y="runs", data=df_manhattan, color="blue")
#     plt.xlabel("Overs")
#     plt.ylabel("Runs")
#     plt.title("Manhattan Chart - Runs per Over")
#     st.pyplot(plt)

# # Worm Chart (Total Runs over Time)
# def plot_worm_chart(df_statistics):
#     if df_statistics.empty:
#         st.warning("No statistics available.")
#         return
#     df_worm = pd.DataFrame(df_statistics.iloc[0]["statistics"]["worm"])
#     plt.figure(figsize=(8, 4))
#     sns.lineplot(x="over", y="runs", data=df_worm, marker="o", color="green")
#     plt.xlabel("Overs")
#     plt.ylabel("Total Runs")
#     plt.title("Worm Chart - Runs Progression")
#     st.pyplot(plt)

# # Run Rate Progression
# def plot_run_rate_chart(df_statistics):
#     if df_statistics.empty:
#         st.warning("No statistics available.")
#         return
#     df_runrate = pd.DataFrame(df_statistics.iloc[0]["statistics"]["runrates"])
#     plt.figure(figsize=(8, 4))
#     sns.lineplot(x="over", y="runrate", data=df_runrate, marker="o", color="red")
#     plt.xlabel("Overs")
#     plt.ylabel("Run Rate")
#     plt.title("Run Rate Progression")
#     st.pyplot(plt)

# # Wagon Wheel (Shot Placement)
# def plot_wagon_wheel(df_wagon):
#     if df_wagon.empty:
#         st.warning("No wagon data available.")
#         return
#     df_wagons = pd.DataFrame(df_wagon.iloc[0]["wagons"], columns=["batsman_id", "bowler_id", "over", "bat_run", "team_run", "x", "y", "zone_id", "event_name", "unique_over"])
#     plt.figure(figsize=(6, 6))
#     sns.scatterplot(x=df_wagons["x"], y=df_wagons["y"], hue=df_wagons["event_name"], palette="tab10", edgecolor="black")
#     plt.xlabel("X Position")
#     plt.ylabel("Y Position")
#     plt.title("Wagon Wheel - Shot Placement")
#     st.pyplot(plt)

# # Partnership Analysis
# def plot_partnership_chart(df_statistics):
#     if df_statistics.empty:
#         st.warning("No partnership data available.")
#         return
    
#     # Ensure partnerships data exists
#     partnerships_data = df_statistics.iloc[0].get("statistics", {}).get("partnership", [])
#     if not partnerships_data:
#         st.warning("No valid partnership data found.")
#         return

#     # Convert into DataFrame
#     df_partnerships = pd.DataFrame(partnerships_data)

#     # Ensure necessary fields exist
#     if "batsmen" not in df_partnerships or "runs" not in df_partnerships:
#         st.warning("Missing required fields in partnership data.")
#         return

#     # Convert batsmen info into a readable format
#     df_partnerships["batsmen_pair"] = df_partnerships["batsmen"].apply(lambda x: f"{x[0]['batsman_id']} & {x[1]['batsman_id']}" if len(x) == 2 else "Unknown")

#     # Plot
#     plt.figure(figsize=(8, 4))
#     sns.barplot(x=df_partnerships["batsmen_pair"], y=df_partnerships["runs"], color="orange")
#     plt.xticks(rotation=45)
#     plt.xlabel("Partnerships")
#     plt.ylabel("Runs")
#     plt.title("Partnership Contributions")
#     st.pyplot(plt)


# # Dismissal Breakdown
# def plot_dismissal_chart(df_statistics):
#     if df_statistics.empty:
#         st.warning("No dismissal data available.")
#         return
#     df_dismissals = pd.DataFrame(df_statistics.iloc[0]["statistics"]["wickets"])
#     plt.figure(figsize=(8, 4))
#     sns.barplot(x="name", y="value", data=df_dismissals, color="purple")
#     plt.xlabel("Dismissal Type")
#     plt.ylabel("Count")
#     plt.title("Types of Wickets")
#     st.pyplot(plt)

# # Generate PDF Report
# def generate_pdf(df_statistics):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", "B", 16)
#     pdf.cell(200, 10, "Cricket Match Report", ln=True, align="C")
#     pdf.ln(10)

#     pdf.set_font("Arial", "B", 12)
#     pdf.cell(200, 10, "Match Summary", ln=True)
    
#     for _, row in df_statistics.iterrows():
#         pdf.set_font("Arial", size=10)
#         pdf.cell(200, 10, f"{row['name']} - {row['runs']} Runs, {row['overs']} Overs, {row['wickets']} Wickets", ln=True)
    
#     pdf.output("match_report.pdf")
#     return "match_report.pdf"

# # Streamlit UI
# st.title("🏏 Cricket Match Analytics Dashboard")

# # Load Matches
# matches = load_match_list()
# match_id = st.selectbox("Select Match ID", list(matches.keys()), format_func=lambda x: matches[x])

# if match_id:
#     base_url = st.secrets["api_url"]
#     all_data = fetch_match_data(match_id, base_url)
#     df_commentary, df_wagon, df_statistics = create_dataframes(all_data)

#     # Generate Visualizations
#     plot_manhattan_chart(df_statistics)
#     plot_worm_chart(df_statistics)
#     plot_run_rate_chart(df_statistics)
#     plot_wagon_wheel(df_wagon)
#     plot_partnership_chart(df_statistics)
#     plot_dismissal_chart(df_statistics)

#     # Generate Report Button
#     if st.button("📄 Generate PDF Report"):
#         pdf_file = generate_pdf(df_statistics)
#         with open(pdf_file, "rb") as file:
#             st.download_button("Download Report", file, "match_report.pdf", mime="application/pdf")
# +++++++++++++V1

# import streamlit as st
# import pandas as pd
# import json
# import matplotlib.pyplot as plt
# import seaborn as sns
# from fpdf import FPDF
# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer

# # Download Sentiment Analysis Tool
# nltk.download('vader_lexicon')
# sia = SentimentIntensityAnalyzer()

# # Load JSON Data
# def load_data(uploaded_file):
#     """Reads the uploaded JSON file from Streamlit"""
#     data = json.load(uploaded_file)  # Directly load JSON from file-like object
#     return data['response']

# # Extract key details
# def get_match_summary(data):
#     inning = data['inning']
#     return {
#         "Team": inning['name'],
#         "Total Runs": inning['scores'],
#         "Overs": inning['scores_full'].split('(')[-1].replace(")", ""),
#         "Wickets": len([b for b in inning['batsmen'] if b['how_out']])
#     }

# # Get Top Players
# def get_top_batsman(data):
#     batsmen = data['inning']['batsmen']
#     df = pd.DataFrame(batsmen)
#     df['runs'] = df['runs'].astype(int)
#     df['strike_rate'] = df['strike_rate'].astype(float)
#     return df[['name', 'runs', 'balls_faced', 'strike_rate']].sort_values(by='runs', ascending=False)

# # Get Top Bowlers
# def get_top_bowler(data):
#     bowlers = data['inning']['bowlers']
#     df = pd.DataFrame(bowlers)
#     df['wickets'] = df['wickets'].astype(int)
#     df['econ'] = df['econ'].astype(float)
#     return df[['name', 'overs', 'runs_conceded', 'wickets', 'econ']].sort_values(by='wickets', ascending=False)

# # Sentiment Analysis on Commentary
# def get_sentiment_analysis(data):
#     comments = [c['commentary'] for c in data['commentaries'] if 'commentary' in c]
#     sentiments = [sia.polarity_scores(c)['compound'] for c in comments]
#     return pd.DataFrame({'Commentary': comments, 'Sentiment': sentiments})

# # Generate PDF Report
# def generate_pdf(match_summary, batsmen_df, bowlers_df, sentiment_df):
#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.add_page()
    
#     pdf.set_font("Arial", "B", 16)
#     pdf.cell(200, 10, "Cricket Match Report", ln=True, align="C")
#     pdf.ln(10)

#     pdf.set_font("Arial", "B", 12)
#     pdf.cell(200, 10, "Match Summary", ln=True)
    
#     pdf.set_font("Arial", size=10)
#     for key, value in match_summary.items():
#         pdf.cell(200, 10, f"{key}: {value}", ln=True)

#     pdf.ln(10)
#     pdf.set_font("Arial", "B", 12)
#     pdf.cell(200, 10, "Top Batsmen", ln=True)
    
#     pdf.set_font("Arial", size=10)
#     for index, row in batsmen_df.iterrows():
#         pdf.cell(200, 10, f"{row['name']} - {row['runs']} Runs, SR: {row['strike_rate']}", ln=True)

#     pdf.ln(10)
#     pdf.set_font("Arial", "B", 12)
#     pdf.cell(200, 10, "Top Bowlers", ln=True)
    
#     pdf.set_font("Arial", size=10)
#     for index, row in bowlers_df.iterrows():
#         pdf.cell(200, 10, f"{row['name']} - {row['wickets']} Wickets, Econ: {row['econ']}", ln=True)

#     pdf.ln(10)
#     pdf.set_font("Arial", "B", 12)
#     pdf.cell(200, 10, "Commentary Sentiment Analysis", ln=True)
    
#     pdf.set_font("Arial", size=10)
#     positive = sentiment_df[sentiment_df['Sentiment'] > 0.2].shape[0]
#     negative = sentiment_df[sentiment_df['Sentiment'] < -0.2].shape[0]
    
#     pdf.cell(200, 10, f"Positive Comments: {positive}, Negative Comments: {negative}", ln=True)
    
#     pdf.output("cricket_report.pdf")
#     return "cricket_report.pdf"

# # Streamlit UI
# st.title("🏏 Cricket Match Report Generator")

# uploaded_file = st.file_uploader("Upload JSON match data", type=['json'])

# if uploaded_file:
#     data = load_data(uploaded_file)

#     st.subheader("📊 Match Summary")
#     match_summary = get_match_summary(data)
#     st.json(match_summary)

#     st.subheader("🏏 Top Batsmen")
#     batsmen_df = get_top_batsman(data)
#     st.dataframe(batsmen_df)

#     st.subheader("🎯 Top Bowlers")
#     bowlers_df = get_top_bowler(data)
#     st.dataframe(bowlers_df)

#     st.subheader("📢 Sentiment Analysis on Commentary")
#     sentiment_df = get_sentiment_analysis(data)
#     st.dataframe(sentiment_df)

#     # Generate PDF
#     if st.button("📄 Generate Report"):
#         pdf_file = generate_pdf(match_summary, batsmen_df, bowlers_df, sentiment_df)
#         with open(pdf_file, "rb") as file:
#             st.download_button("Download Report", file, "match_report.pdf", mime="application/pdf")

# ========

# import streamlit as st
# import pandas as pd
# # import matplotlib.pyplot as plt
# import seaborn as sns

# # Load your dataset (replace 'your_dataset.csv' with the actual file path)
# df = pd.read_csv(r'C:\Users\USER\Downloads\Telegram Desktop\deliveries_updated_mens_ipl.csv')


# # Get unique player names from the dataset
# player_names = df['batsman'].unique()
# st.set_page_config(layout="centered")  # Set the layout to wide for full screen
# # st.set_page_config(physical_keyboard_only=True)
# st.title("See your Favorite IPL Players: By the Numbers and Stats")

# st.write(
#     "Welcome to the IPL Cricket Stats Explorer - your go-to destination for unraveling the fascinating world of IPL player performances! 🏏✨\n\n"
#     "Dive into the heart of the game as we bring you closer to your favorite cricket stars through an interactive and user-friendly experience. Discover the stories behind every boundary, every six, and every wicket with our intuitive interface.\n\n"
#     "**Key Features:**\n\n"
#     "📊 **Player Stats at Your Fingertips:** Explore the latest stats of IPL players, from total runs and averages to boundary percentages and dismissal analyses.\n\n"
#     "🔍 **Match-by-Match Breakdown:** Uncover the excitement of the last **N** matches, with in-depth insights into each player's runs, strike rates, and more.\n\n"
#     "⚡ **Dynamic Slider:** Tailor your experience with a user-friendly slider to easily select the number of matches you want to explore.\n\n"
#     "Get ready for an adventure into the world of cricket statistics – where every run counts, and every player shines! 🌐🏆"
# )

# col_a,col_b = st.columns(2)
# # Dropdown to select a player
# selected_player = col_a.selectbox("Select a Player", player_names)

# # Slider to set the match count
# default_match_count = 10

# match_count = col_b.slider("Select the number of matches", 1, 50, default_match_count)

# # Filter data for the selected player
# player_data = df[df['batsman'] == selected_player].sort_values(by ='matchId',ascending=False)

# # Function to calculate statistics
# def calculate_statistics(data):
#     last_10_matches_score = data.groupby(['batsman', 'matchId', 'date', 'batting_team', 'bowling_team']).agg({'batsman_runs': 'sum', 'ball': 'count','dismissal_kind': 'first'})
#     last_10_matches_score.columns = ['total_runs', 'balls_faced', 'dismissal_kind']
#     # Sort by matchId in descending order
#     last_10_matches_score = last_10_matches_score.sort_values(by='matchId', ascending=False)
#     last_10_matches_score = last_10_matches_score.head(match_count)
#     unique_match_ids = data['matchId'].unique()[:match_count]
#     last_10_matches = data[data['matchId'].isin(unique_match_ids)]
#     # Calculate additional statistics
#     total_runs = last_10_matches['batsman_runs'].sum()
#     average = total_runs / match_count
#     strike_rate = (total_runs / last_10_matches.shape[0]) * 100
#     boundary_percentage = (last_10_matches[last_10_matches['batsman_runs'].isin([4, 6])].shape[0] / match_count) * 100
    
#     fifty_count = last_10_matches_score[((last_10_matches_score['total_runs'] >= 50) & (last_10_matches_score['total_runs']<100))].shape[0]
#     hundred_count = last_10_matches_score[last_10_matches_score['total_runs'] >= 100].shape[0]
#     runs_per_dismissal = last_10_matches_score['total_runs'].sum() / last_10_matches_score.shape[0]
#     fours_count = last_10_matches[last_10_matches['batsman_runs'].isin([4])].shape[0]
#     sixes_count = last_10_matches[last_10_matches['batsman_runs'].isin([6])].shape[0]
#     dots_percentage = (last_10_matches[last_10_matches['batsman_runs'] == 0].shape[0] / last_10_matches.shape[0]) * 100
#     ones_count = last_10_matches[last_10_matches['batsman_runs'] == 1].shape[0]
#     twos_count = last_10_matches[last_10_matches['batsman_runs'] == 2].shape[0]
#     threes_count = last_10_matches[last_10_matches['batsman_runs'] == 3].shape[0]
#     dismissal_analysis = player_data.groupby('dismissal_kind').size().reset_index(name='count').sort_values(by='count', ascending=False)

#     return dismissal_analysis,total_runs,average,strike_rate,runs_per_dismissal,fifty_count, hundred_count,boundary_percentage, fours_count, sixes_count, dots_percentage, ones_count, twos_count, threes_count, last_10_matches_score

# # Calculate statistics for the selected player
# dismissal_analysis,total_runs,average,strike_rate,runs_per_dismissal,fifty_count, hundred_count,boundary_percentage, fours_count, sixes_count, dots_percentage, ones_count, twos_count, threes_count, last_10_matches_score = calculate_statistics(player_data)

# col1, col2 = st.columns(2)
# # Display the statistics


# st.title(f"{selected_player}'s Last {match_count} Matches Statistics")
# col1.write(f"**Total Runs**: {total_runs}")
# col1.write(f"**Runs per Dismissal**: {runs_per_dismissal:.2f}")
# col1.write(f"**Average**: {average:.2f}")
# col1.write(f"**Strike Rate**: {strike_rate:.2f}%")
# col1.write(f"**Dots Percentage**: {dots_percentage:.2f}%")
# col1.write(f"**Boundary Percentage**: {boundary_percentage:.2f}%")
# st.write(f"**dismissal analysis**: {dismissal_analysis}")
# col2.write(f"**1s**: {ones_count}")
# col2.write(f"**2s**: {twos_count}")
# col2.write(f"**3s**: {threes_count}")
# col2.write(f"**4s**: {fours_count}")
# col2.write(f"**6s**: {sixes_count}")
# col2.write(f"**50s**: {fifty_count}")
# col2.write(f"**100s**: {hundred_count}")


# # Display the last 10 matches data for the selected player
# st.subheader(f"Last {match_count} Matches Data for {selected_player}:")
# st.write(last_10_matches_score.head(match_count))

# # Bar chart for Runs Distribution
# fig, ax = plt.subplots()
# sns.histplot(last_10_matches_score['total_runs'], bins=range(0, 101, 10), kde=False, ax=ax)
# ax.set_title(f"{selected_player}'s Runs Distribution in Last {match_count} Matches")
# ax.set_xlabel("Runs")
# ax.set_ylabel("Frequency")
# st.pyplot(fig)

# #pie chart
# fig_pie_chart, ax_pie_chart = plt.subplots()
# labels = ['1s', '2s', '3s', '4s', '6s']
# sizes = [ones_count, twos_count, threes_count, fours_count, sixes_count]
# ax_pie_chart.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3"))
# ax_pie_chart.set_title(f"{selected_player}'s Runs Distribution (1's to 6's) in Last 10 Matches")
# st.pyplot(fig_pie_chart)

# st.text("Hope you have got some intresting stats")
# st.text("Thank You - by Anish S")
