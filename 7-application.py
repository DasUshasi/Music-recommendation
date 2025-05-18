from dash import dcc, html, Input, Output, Dash, dash_table
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
from scipy.spatial.distance import cdist
from collections import Counter
import plotly.graph_objects as go



''' datasets '''
song_details = pd.read_csv("datasets/song_details.csv")
data = pd.read_csv("datasets/songs_features.csv")

for i, song in song_details.iterrows():
    song_details.at[i, "name_link"] = f"[Go to song 🔗](https://open.spotify.com/track/{song['id']})"



''' function to get index of the songs '''
def get_indices(songs):
    indices = []
    for song in songs:
        idx = song_details[song_details['name'] == song].index
        for i in idx:
            indices.append(i)
    return indices



''' model and recommendor function '''
model = KMeans(n_clusters=7, random_state=42)
model.fit(data)
clusters = model.predict(data)

''' function that recommends songs '''
def recommend(songs):
    # get the indices of the songs
    indices = []
    for song in songs:
        idx = song_details[song_details['name'] == song].index[0]
        indices.append(idx)
    
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



''' Dash app '''
app = Dash(__name__)

''' Layout of the app '''
app.layout = html.Div([
    html.H1("Tune Finder", style={"textAlign": "center"}),
    html.H3("Find songs you'll love by analyzing the music you already enjoy, with recommendations tailored to your unique taste.", style={"textAlign": "center"}),
    
    # Dropdown for selecting songs
    html.H2("Select Songs:", 
            style={'width': '97vw', 'margin': '5px auto 7px auto', 'textAlign': 'left'}),
    dcc.Dropdown(
        id='song-selector',
        options=[{"label": f"{song['name']} [{song['artist']}]", "value": song['name']} 
                 for _, song in song_details.iterrows()],
        multi=True,
        style={'fontSize': '1.1rem', 'width': '97vw', 'margin': '0 auto'}
    ),
    
    # Div to display details of the selected songs
    html.Div(id='selected-song-details', style={"marginTop": "20px"}),
    html.Div([
        dcc.Graph(id='genre-selected')
    ], id='genre-selected-div', style={'display': 'none'}),

    # Div to display recommendations
    html.Div(id='recommendations', style={"marginTop": "20px"}),
    html.Div([
        dcc.Graph(id='genre-recommend-pie')
    ], id='genre-recommend-pie-div', style={'display': 'none'}),
    html.Div([
        dcc.Graph(id='genre-recommend-line', style={'width': '80vw'})
    ], id='genre-recommend-line-div', style={'display': 'none'})
], style={"overflow-x": "hidden", "height": "100vh"})

''' Callback to show details of selected songs '''
@app.callback(
    [Output('selected-song-details', 'children'),
     Output('genre-selected-div', 'style'),
     Output('genre-selected', 'figure')],
    Input('song-selector', 'value')
)
def update_selected_songs(selected_songs):
    if not selected_songs:
        return "No songs selected.", {'display': 'none'}, {}
    
    indices = get_indices(selected_songs)
    song_data = song_details.iloc[indices].to_dict('records')

    selected_genres = []
    for song in song_data:
        selected_genres.extend(song['genre'].split(', '))
    selected_genres = Counter(selected_genres)

    heading = html.H1("Selected Songs", style={"textAlign": "center"})

    # Display the selected songs in a table format
    table = dash_table.DataTable(
        id="selected_table",
        columns=[
            {'name': "Song Name", 'id': "name"},
            {'name': "Artist", 'id': "artist"},
            {'name': "Genre", 'id': "genre"},
            {'name': "Release year", 'id': "release_year"}
        ],
        data=song_data,
        style_table={'width': '85vw', 'margin': '0 auto'},
        style_header={'fontSize': '1.3rem', 'fontWeight': 'bold', 'textAlign': 'center'},
        style_cell={'textAlign': 'center', 'fontSize': '1.2rem',
                    'whiteSpace': 'normal', 'wordBreak': 'break-word'}
    )

    # Pie chart to show distribution of genres of the songs selected
    genre_pie_chart = px.pie(names=selected_genres.keys(), 
                             values=selected_genres.values(), 
                             title="Selected Songs Genre Distribution")
    genre_pie_chart.update_layout(
        title_x=0.5,             
        title_xanchor='center'  
    )

    return [heading, table], {'display': 'flex', 'width': '100vw', "justifyContent": "center"}, genre_pie_chart

''' Recommendation callback '''
@app.callback(
    [Output('recommendations', 'children'),
     Output('genre-recommend-pie-div', 'style'),
     Output('genre-recommend-line-div', 'style'),
     Output('genre-recommend-pie', 'figure'),
     Output('genre-recommend-line', 'figure')],
    Input('song-selector', 'value')
)
def recommend_songs(selected_songs):
    if not selected_songs:
        return "Please select at least one song to get recommendations.", {'display': 'none'}, {'display': 'none'}, {}, {}
    
    recommended_songs, most_similar = recommend(selected_songs)
    recommended_songs = recommended_songs.to_dict('records')

    recommend_genres = []
    recommend_years = []
    for song in recommended_songs:
        recommend_genres.extend(song['genre'].split(', '))
        recommend_years.append(song['release_year'])
    recommend_genres = Counter(recommend_genres)
    recommend_years = Counter(recommend_years)

    style_data_conditional = [
        {
            'if': {'filter_query': f"'{item}' = {{name}}"},
            'backgroundColor': 'lightyellow'
        } 
        for item in most_similar] + [
            {
                'if': {'column_id': 'name_link'},
                'width': '13rem'
            }
        ]

    heading = html.H1("Recommendations", style={"textAlign": "center"})

    # Display the recommended songs in a table format
    table = dash_table.DataTable(
        id="selected_table",
        data=recommended_songs,
        columns=[
            {'name': "Song Name", 'id': "name"}, 
            {'name': "Artist", 'id': "artist"},
            {'name': "Genre", 'id': "genre"},
            {'name': "Release year", 'id': "release_year"},
            {'name': "Link", 'id': "name_link", "presentation": "markdown"}
        ],
        style_table={'width': '85vw', 'margin': '0 auto'},
        style_header={'fontSize': '1.3rem', 'fontWeight': 'bold', 'textAlign': 'center'},
        style_cell={'textAlign': 'center', 'fontSize': '1.2rem',
                    'whiteSpace': 'normal', 'margin-bottom':'0'}, 
        style_data_conditional=style_data_conditional,
        css=[dict(selector= "p", rule= "text-align: center")]
    )

    # Pie chart to show distribution of genres of the songs selected
    genre_pie_chart = px.pie(names=recommend_genres.keys(), 
                             values=recommend_genres.values(), 
                             title="Recommended Songs Genre Distribution")
    genre_pie_chart.update_layout(
        title_x=0.5,             
        title_xanchor='center'  
    )
    
    # Histogram graph to show years when the recommended songs were released
    year_hist = go.Figure(data=go.Bar(x=list(recommend_years.keys()), 
                                        y=list(recommend_years.values())))
    year_hist.update_layout(
        title="Recommended Songs Release Year Distribution",
        xaxis=dict(
            tickvals=list(recommend_years.keys()),      
            ticktext=list(recommend_years.keys()),
            title='Release Year'        
        ),
        yaxis=dict(title='Frequency'),
        title_x=0.5,             
        title_xanchor='center'
    )
    
    return [heading, table], {'display': 'flex', 'width': '100vw', "justifyContent": "center"}, {'display': 'flex', 'width': '100vw', "justifyContent": "center"}, genre_pie_chart, year_hist

app.run(debug=True)