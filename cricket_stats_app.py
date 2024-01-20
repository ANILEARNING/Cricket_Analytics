import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv(r'C:\Users\USER\Downloads\Telegram Desktop\deliveries_updated_mens_ipl.csv')


# Get unique player names from the dataset
player_names = df['batsman'].unique()
st.set_page_config(layout="centered")  # Set the layout to wide for full screen
# st.set_page_config(physical_keyboard_only=True)
st.title("See your Favorite IPL Players: By the Numbers and Stats")

st.write(
    "Welcome to the IPL Cricket Stats Explorer - your go-to destination for unraveling the fascinating world of IPL player performances! 🏏✨\n\n"
    "Dive into the heart of the game as we bring you closer to your favorite cricket stars through an interactive and user-friendly experience. Discover the stories behind every boundary, every six, and every wicket with our intuitive interface.\n\n"
    "**Key Features:**\n\n"
    "📊 **Player Stats at Your Fingertips:** Explore the latest stats of IPL players, from total runs and averages to boundary percentages and dismissal analyses.\n\n"
    "🔍 **Match-by-Match Breakdown:** Uncover the excitement of the last **N** matches, with in-depth insights into each player's runs, strike rates, and more.\n\n"
    "⚡ **Dynamic Slider:** Tailor your experience with a user-friendly slider to easily select the number of matches you want to explore.\n\n"
    "Get ready for an adventure into the world of cricket statistics – where every run counts, and every player shines! 🌐🏆"
)

col_a,col_b = st.columns(2)
# Dropdown to select a player
selected_player = col_a.selectbox("Select a Player", player_names)

# Slider to set the match count
default_match_count = 10

match_count = col_b.slider("Select the number of matches", 1, 50, default_match_count)

# Filter data for the selected player
player_data = df[df['batsman'] == selected_player].sort_values(by ='matchId',ascending=False)

# Function to calculate statistics
def calculate_statistics(data):
    last_10_matches_score = data.groupby(['batsman', 'matchId', 'date', 'batting_team', 'bowling_team']).agg({'batsman_runs': 'sum', 'ball': 'count','dismissal_kind': 'first'})
    last_10_matches_score.columns = ['total_runs', 'balls_faced', 'dismissal_kind']
    # Sort by matchId in descending order
    last_10_matches_score = last_10_matches_score.sort_values(by='matchId', ascending=False)
    last_10_matches_score = last_10_matches_score.head(match_count)
    unique_match_ids = data['matchId'].unique()[:match_count]
    last_10_matches = data[data['matchId'].isin(unique_match_ids)]
    # Calculate additional statistics
    total_runs = last_10_matches['batsman_runs'].sum()
    average = total_runs / match_count
    strike_rate = (total_runs / last_10_matches.shape[0]) * 100
    boundary_percentage = (last_10_matches[last_10_matches['batsman_runs'].isin([4, 6])].shape[0] / match_count) * 100
    
    fifty_count = last_10_matches_score[((last_10_matches_score['total_runs'] >= 50) & (last_10_matches_score['total_runs']<100))].shape[0]
    hundred_count = last_10_matches_score[last_10_matches_score['total_runs'] >= 100].shape[0]
    runs_per_dismissal = last_10_matches_score['total_runs'].sum() / last_10_matches_score.shape[0]
    fours_count = last_10_matches[last_10_matches['batsman_runs'].isin([4])].shape[0]
    sixes_count = last_10_matches[last_10_matches['batsman_runs'].isin([6])].shape[0]
    dots_percentage = (last_10_matches[last_10_matches['batsman_runs'] == 0].shape[0] / last_10_matches.shape[0]) * 100
    ones_count = last_10_matches[last_10_matches['batsman_runs'] == 1].shape[0]
    twos_count = last_10_matches[last_10_matches['batsman_runs'] == 2].shape[0]
    threes_count = last_10_matches[last_10_matches['batsman_runs'] == 3].shape[0]
    dismissal_analysis = player_data.groupby('dismissal_kind').size().reset_index(name='count').sort_values(by='count', ascending=False)

    return dismissal_analysis,total_runs,average,strike_rate,runs_per_dismissal,fifty_count, hundred_count,boundary_percentage, fours_count, sixes_count, dots_percentage, ones_count, twos_count, threes_count, last_10_matches_score

# Calculate statistics for the selected player
dismissal_analysis,total_runs,average,strike_rate,runs_per_dismissal,fifty_count, hundred_count,boundary_percentage, fours_count, sixes_count, dots_percentage, ones_count, twos_count, threes_count, last_10_matches_score = calculate_statistics(player_data)

col1, col2 = st.columns(2)
# Display the statistics


st.title(f"{selected_player}'s Last {match_count} Matches Statistics")
col1.write(f"**Total Runs**: {total_runs}")
col1.write(f"**Runs per Dismissal**: {runs_per_dismissal:.2f}")
col1.write(f"**Average**: {average:.2f}")
col1.write(f"**Strike Rate**: {strike_rate:.2f}%")
col1.write(f"**Dots Percentage**: {dots_percentage:.2f}%")
col1.write(f"**Boundary Percentage**: {boundary_percentage:.2f}%")
st.write(f"**dismissal analysis**: {dismissal_analysis}")
col2.write(f"**1s**: {ones_count}")
col2.write(f"**2s**: {twos_count}")
col2.write(f"**3s**: {threes_count}")
col2.write(f"**4s**: {fours_count}")
col2.write(f"**6s**: {sixes_count}")
col2.write(f"**50s**: {fifty_count}")
col2.write(f"**100s**: {hundred_count}")


# Display the last 10 matches data for the selected player
st.subheader(f"Last {match_count} Matches Data for {selected_player}:")
st.write(last_10_matches_score.head(match_count))

# Bar chart for Runs Distribution
fig, ax = plt.subplots()
sns.histplot(last_10_matches_score['total_runs'], bins=range(0, 101, 10), kde=False, ax=ax)
ax.set_title(f"{selected_player}'s Runs Distribution in Last {match_count} Matches")
ax.set_xlabel("Runs")
ax.set_ylabel("Frequency")
st.pyplot(fig)

#pie chart
fig_pie_chart, ax_pie_chart = plt.subplots()
labels = ['1s', '2s', '3s', '4s', '6s']
sizes = [ones_count, twos_count, threes_count, fours_count, sixes_count]
ax_pie_chart.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3"))
ax_pie_chart.set_title(f"{selected_player}'s Runs Distribution (1's to 6's) in Last 10 Matches")
st.pyplot(fig_pie_chart)

st.text("Hope you have got some intresting stats")
st.text("Thank You - by Anish S")
