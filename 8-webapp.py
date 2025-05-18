import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import requests
from streamlit.components.v1 import html
from collections import Counter
from dotenv import load_dotenv
import os

load_dotenv()



# spotify api 
url = "https://accounts.spotify.com/api/token"
headers = {
    "Content-Type": "application/x-www-form-urlencoded"
}
data = {
    "grant_type": "client_credentials",
    "client_id": os.getenv("client_id"),
    "client_secret": os.getenv("client_secret")
}

response = requests.post(url, headers=headers, data=data)
access_token = response.json()["access_token"]


# datasets 
song_details = pd.read_csv("datasets/song_details.csv")
data = pd.read_csv("datasets/songs_features.csv")

for i, song in song_details.iterrows():
    song_details.at[i, "link"] = f"https://open.spotify.com/track/{song['id']}"



# function to get index of the songs 
def get_indices(songs):
    indices = []
    for song in songs:
        idx = song_details[song_details['name'] == song].index
        indices.extend(idx)
    return indices



# model and recommendor function 
model = KMeans(n_clusters=7, random_state=42)
model.fit(data)
clusters = model.predict(data)

# function that recommends songs 
def recommend(songs, indices):
    # get the features of the songs 
    features = data.iloc[indices]

    # get clusters of the input songs
    input_clusters = model.predict(features)

    # get the closest songs
    closest_songs = []
    top = []
    for i, cluster in enumerate(input_clusters):
        # Find all songs in the same cluster
        cluster_indices = np.where(clusters == cluster)[0]
        cluster_songs = data.iloc[cluster_indices]
    
        # Calculate distances between the input song and songs in the same cluster
        distances = cdist([features.iloc[i]], cluster_songs, metric='cosine')
    
        # Sort songs by distance (ascending) and select the closest ones
        closest_indices = cluster_indices[np.argsort(distances[0])][: 7]  # Top closest songs
        closest_songs.extend(closest_indices)
        top.append(closest_indices)
        
    recommended_songs = song_details.iloc[list(set(closest_songs))]
    
    # dont take the songs that were given as input
    recommended_songs = recommended_songs[~recommended_songs['name'].isin(songs)]

    # find the most similar song for each song
    most_similar = []
    for i, cluster in enumerate(input_clusters):
        top_songs = song_details.iloc[top[i]]
        top_name = top_songs[~top_songs['name'].isin(songs)].iloc[0]['name']
        most_similar.append(top_name)

    return recommended_songs, list(set(most_similar))



# streamlit application

# css
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        header {visibility: hidden;}

        footer {visibility: hidden;}

        .block-container {
            padding-top: 0px; 
            margin-top: 0px; 
        }
    </style>
    """,
    unsafe_allow_html=True
)


css = """
    <style>
        .container {
            display: flex;
            justify-content: space-around;
            gap: 0.8rem;
            width: 90vw;
            flex-wrap: wrap;
            padding: 1rem;
        }

        .card {
            text-align: center;
            border-radius: 0.5rem;
            width: 22rem;
            background-color: #EFF2FF;
            padding: 12px 0;
            color: #050A2C;
        }
    </style>
"""

# script to dynamically change height
script = """
<script>
    function adjustIframeHeight() {
        const iframe = window.frameElement;
        if (iframe) {
            iframe.style.height = document.body.scrollHeight + "px"; // Adjust iframe height
        }
    }

    // Adjust height when the iframe content is fully loaded
    window.addEventListener("load", adjustIframeHeight);

    // Adjust height when the window is resized
    window.addEventListener("resize", adjustIframeHeight);

</script>
"""

# Generate dynamic HTML for cards from the dataframe
def generate_cards(song, most_similar):
    card_html = ""
    for _, row in song.iterrows():
        url = f"https://api.spotify.com/v1/tracks/{row['id']}"
        response = requests.get(url, headers={"Authorization": f"Bearer {access_token}"})
        json = response.json()
        image_url = ""
        if json['album']['images']:
            image_url = json['album']['images'][0]['url']

        card_html += f"""
        <div class="card">
            <img src="{image_url}" alt="Spotify Album Cover" style="width: 250px;">
            <p style="font-size: 1rem; height: 1rem">{"⭐ Most similar" if most_similar and row['name'] in most_similar else ""}</p>
            <p style="font-size: 1.5rem; font-weight: bold">{row['name']}</p>
            <p style="font-size: 1.2rem">Artist(s): {row['artist']}<p>
            <p style="font-size: 1.2rem">Genre: {row['genre']}</p>
            <p style="font-size: 1.2rem">Release year: {row['release_year']}</p>
            <a href="{row['link']}" target="_blank" style="font-size: 1.2rem; text-decoration: none;">Go to song 🔗</a>
        </div>
        """
    return card_html

st.markdown(
    '<p style="font-size: 4rem; color: #200087; font-family: Comic Sans MS, cursive; text-align: center;">'
    'Tune Finder'
    '</p>',
    unsafe_allow_html=True
)
st.markdown("<h5 style='text-align: center'>Find songs you'll love by analyzing the music you already enjoy, with recommendations tailored to your unique taste.</h5>", unsafe_allow_html=True)
st.markdown("<h3>Select songs:</h3>", unsafe_allow_html=True)

# dropdown
value_to_label = {f"{row['name']}": f"{row['name']}[{row['artist']}]" for _, row in song_details.iterrows()}
options = list(value_to_label.keys())
selected_songs = st.multiselect(
    "Song options: ",
    options=options,
    format_func=lambda x: value_to_label[x]
)

# create a button which will be pressed for recommendations
if st.button("Get Recommendations"):
    indices = []
    indices = get_indices(selected_songs)
    selected_song_data = song_details.iloc[indices]

    # display selected songs
    st.markdown("<h3>Selected songs: </h3>", unsafe_allow_html=True)
    html_code = css

    html_code += f"""
        <div class="container">
            {generate_cards(selected_song_data, None)}
        </div>
    """

    html_code += script
    html(html_code, height=500)

    selected_genres = []
    for i, song in selected_song_data.iterrows():
        selected_genres.extend(song['genre'].split(', '))

    # Pie chart to show distribution of genres of the songs selected
    selected_genres = Counter(selected_genres)

    st.markdown("<h4>Distribution of genres of selected songs: </h4>", unsafe_allow_html=True)
    genre_pie_chart = px.pie(names=selected_genres.keys(), 
                             values=selected_genres.values())
    st.plotly_chart(genre_pie_chart)

    # get recommendations
    recommended_songs, most_similar = recommend(selected_songs, indices)

    # display recommended songs
    st.markdown("<h3>Recommended songs: </h3>", unsafe_allow_html=True)
    html_code = css

    html_code += f"""
        <div class="container">
            {generate_cards(recommended_songs, most_similar)}
        </div>
    """

    html_code += script
    html(html_code, height=500)

    recommend_genres = []
    recommend_years = []
    for i, song in recommended_songs.iterrows():
        recommend_genres.extend(song['genre'].split(', '))
        recommend_years.append(song['release_year'])

    # Pie chart to show distribution of genres of the songs selected
    recommend_genres = Counter(recommend_genres)

    st.markdown("<h4>Distribution of genres of recommended songs: </h4>", unsafe_allow_html=True)
    genre_pie_chart = px.pie(names=recommend_genres.keys(), 
                             values=recommend_genres.values())
    st.plotly_chart(genre_pie_chart)

    # Histogram graph to show years when the recommended songs were released
    recommend_years = Counter(recommend_years)

    st.markdown("<h4>Distribution of release years of recommended songs: </h4>", unsafe_allow_html=True)
    year_hist = go.Figure(data=go.Bar(x=list(recommend_years.keys()), 
                                        y=list(recommend_years.values())))
    years = list(recommend_years.keys())
    minYear = min(years)
    maxYear = max(years)
    yearRange = list(range(minYear, maxYear + 1, 1))
    year_hist.update_layout(
        xaxis=dict(
            tickvals=yearRange,      
            ticktext=yearRange,
            title='Release Year'        
        ),
        yaxis=dict(title='Frequency')
    )
    st.plotly_chart(year_hist)

