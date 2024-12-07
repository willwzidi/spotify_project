import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import requests
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import numpy as np

# Spotify API credentials
CLIENT_ID = 'fc668f7d3ed94d9f9d0f4f8e9c8af9ad'
CLIENT_SECRET = 'feb9efe5c0224e92a44c52e22aa5b9a2'

# Spotify API Helper Functions
def get_spotify_token():
    url = "https://accounts.spotify.com/api/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials"}
    auth = (CLIENT_ID, CLIENT_SECRET)
    response = requests.post(url, headers=headers, data=data, auth=auth)
    if response.status_code == 200:
        response_data = response.json()
        return response_data.get("access_token")
    else:
        print(f"Error fetching token: {response.status_code} - {response.text}")
        return None

def search_podcasts(query, token):
    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": "show", "limit": 10}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        results = response.json()["shows"]["items"]

        # Filter and sort results based on relevance to the query
        query_lower = query.lower()
        filtered_results = sorted(
            results,
            key=lambda x: query_lower in x["name"].lower() or query_lower in x["description"].lower(),
            reverse=True,
        )
        return filtered_results
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

def generate_clustering_data(podcasts):
    """Generate clustering metrics based on podcast details."""
    data = []
    for podcast in podcasts:
        data.append({
            "Podcast": podcast["name"],
            "Episode": podcast["description"][:50] + "...",  # Truncate description
            "Metric 1": podcast.get("popularity", np.random.uniform(0.5, 1.0)),  # Use popularity if available
            "Metric 2": podcast.get("total_episodes", np.random.randint(5, 100)) / 100,  # Normalize episodes
            "Metric 3": np.random.uniform(0.5, 1.0),  # Random placeholder metric
        })
    return pd.DataFrame(data)

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Layout Components
def create_header():
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand("Spotify Podcast Analysis"),
                dbc.Nav(
                    [
                        dbc.NavItem(
                            dbc.Button(
                                "Podcast Clustering",
                                id="btn-tab-cluster",
                                color="dark",
                                style={"border": "1px solid #ddd"},
                            )
                        ),
                    ],
                    className="ms-auto",
                ),
            ]
        ),
        color="primary",
        dark=True,
    )

def create_sidebar():
    return dbc.Col(
        [
            html.H3("Spotify Tools", className="text-center mt-3"),
            html.Hr(),
            dbc.InputGroup(
                [
                    dbc.InputGroupText("\ud83c\udfa7"),
                    dbc.Input(
                        id="input-podcast",
                        placeholder="Enter Podcast Name",
                        type="text",
                    ),
                ],
                className="mb-3",
            ),
            dbc.Button(
                "Search Podcast",
                id="btn-search-podcast",
                color="success",
                className="w-100 mb-3",
            ),
            html.Div(id="search-results"),
        ],
        width=3,
        style={
            "background-color": "#f8f9fa",
            "height": "100%",
            "padding": "15px",
            "box-shadow": "2px 0px 5px rgba(0, 0, 0, 0.1)",
        },
    )

def create_content():
    return dbc.Col(
        [
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Podcast Clustering",
                        children=[
                            html.Div(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader("Select Podcast"),
                                                        dbc.CardBody(
                                                            dcc.Dropdown(
                                                                id="podcast-dropdown",
                                                                placeholder="Choose a Podcast",
                                                            )
                                                        ),
                                                    ]
                                                ),
                                                width=6,
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader("Metrics Visualization"),
                                                        dbc.CardBody(
                                                            dcc.Graph(id="metrics-visualization")
                                                        ),
                                                    ]
                                                ),
                                                width=6,
                                            ),
                                        ],
                                    ),
                                    html.Br(),
                                    html.H3("Nearest Podcasts/Episodes", className="text-center"),
                                    dash_table.DataTable(
                                        id="nearest-table",
                                        columns=[
                                            {"name": "Podcast", "id": "Podcast"},
                                            {"name": "Episode", "id": "Episode"},
                                            {"name": "Distance", "id": "Distance"},
                                        ],
                                        style_table={"overflowY": "auto", "overflowX": "auto"},
                                        style_cell={"textAlign": "left", "padding": "10px"},
                                    ),
                                ],
                                style={"padding": "20px"},
                            )
                        ],
                    )
                ]
            )
        ],
        width=9,
    )

# Define App Layout
app.layout = dbc.Container(
    [
        create_header(),
        dbc.Row(
            [
                create_sidebar(),
                create_content(),
            ],
            style={"height": "100vh"},
        ),
    ],
    fluid=True,
)

# Callbacks
@app.callback(
    [
        Output("search-results", "children"),
        Output("podcast-dropdown", "options"),
    ],
    [Input("btn-search-podcast", "n_clicks")],
    [State("input-podcast", "value")],
)
def update_podcast_search(n_clicks, query):
    if n_clicks is None or not query:
        return "Please enter a podcast name to search.", []

    try:
        token = get_spotify_token()
        if not token:
            return "Failed to retrieve Spotify token. Please check credentials.", []

        podcasts = search_podcasts(query, token)
        if not podcasts:
            return "No results found. Try a different search term.", []

        clustering_data = generate_clustering_data(podcasts)

        dropdown_options = [
            {"label": podcast["Podcast"], "value": idx}
            for idx, podcast in clustering_data.iterrows()
        ]

        search_list = html.Ul(
            [html.Li(podcast["name"]) for podcast in podcasts],
            style={"padding": "10px", "listStyleType": "none"},
        )
        return search_list, dropdown_options

    except Exception as e:
        print(f"Error during search: {e}")
        return "An error occurred during the search. Please try again later.", []

@app.callback(
    [
        Output("metrics-visualization", "figure"),
        Output("nearest-table", "data"),
    ],
    [Input("podcast-dropdown", "value")],
)
def update_visualization(selected_idx):
    if selected_idx is None:
        return {}, []

    try:
        selected_metrics = df.iloc[selected_idx][["Metric 1", "Metric 2", "Metric 3"]]
        features = df[["Metric 1", "Metric 2", "Metric 3"]].values

        # Calculate distances to other podcasts
        distances = [
            {
                "Podcast": row["Podcast"],
                "Episode": row["Episode"],
                "Distance": euclidean(features[selected_idx], features[idx]),
            }
            for idx, row in df.iterrows() if idx != selected_idx
        ]
        nearest = sorted(distances, key=lambda x: x["Distance"])[:5]

        # Create visualization
        figure = {
            "data": [
                {
                    "x": df["Metric 1"],
                    "y": df["Metric 2"],
                    "text": df["Podcast"],
                    "mode": "markers",
                    "marker": {"size": 12, "color": "blue"},
                },
                {
                    "x": [selected_metrics["Metric 1"]],
                    "y": [selected_metrics["Metric 2"]],
                    "text": ["Selected Podcast"],
                    "mode": "markers",
                    "marker": {"size": 15, "color": "red", "symbol": "star"},
                },
            ],
            "layout": {
                "title": "Podcast Clustering Visualization",
                "xaxis": {"title": "Metric 1"},
                "yaxis": {"title": "Metric 2"},
            },
        }
        return figure, nearest

    except Exception as e:
        print(f"Error during visualization update: {e}")
        return {}, []

# Run the App
if __name__ == "__main__":
    app.run_server(debug=True)
