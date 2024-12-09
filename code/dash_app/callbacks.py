import requests
from dash import Input, Output, State, html
import pandas as pd
import numpy as np
from spotify_helpers import (
    get_spotify_token,
    search_podcasts,
    generate_clustering_data,
    use_pca,
)
from scipy.spatial.distance import euclidean
import time


def register_callbacks(app):
    @app.callback(
        [
            Output("search-results", "children"),
            Output("podcast-dropdown", "options"),
            Output("clustering-data-store", "data"),
        ],
        [Input("btn-search-podcast", "n_clicks")],
        [State("input-podcast", "value")],
    )
    def update_podcast_search(n_clicks, query):
        if n_clicks is None or not query:
            return "Please enter a podcast name to search.", [], None

        try:
            token = get_spotify_token()
            if not token:
                return "Failed to retrieve Spotify token. Please check credentials.", [], None

            podcasts = search_podcasts(query, token)
            if not podcasts:
                return "No results found. Try a different search term.", [], None

            clustering_data = generate_clustering_data(podcasts)
            clustering_data["PCA_Features"] = list(use_pca(clustering_data))

            dropdown_options = [
                {"label": podcast["Podcast"], "value": idx}
                for idx, podcast in clustering_data.iterrows()
            ]

            search_list = html.Ul(
                [html.Li(podcast["name"]) for podcast in podcasts],
                style={"padding": "10px", "listStyleType": "none"},
            )
            return search_list, dropdown_options, clustering_data.to_dict("records")

        except Exception as e:
            print(f"Error during search: {e}")
            return "An error occurred during the search. Please try again later.", [], None

    @app.callback(
        [
            Output("metrics-visualization", "figure"),
            Output("nearest-table", "data"),
        ],
        [
            Input("podcast-dropdown", "value"),
            Input("pca-x-feature-dropdown", "value"),
            Input("pca-y-feature-dropdown", "value"),
        ],
        [State("clustering-data-store", "data")],
    )
    def update_visualization(selected_idx, x_feature, y_feature, clustering_data):
        if selected_idx is None or clustering_data is None or x_feature is None or y_feature is None:
            return {}, []

        try:
            df = pd.DataFrame(clustering_data)
            selected_features = np.array(df.iloc[selected_idx]["PCA_Features"])
            distances = [
                {
                    "Podcast": row["Podcast"],
                    "Episode": row["Episode"],
                    "Distance": euclidean(selected_features, np.array(row["PCA_Features"])),
                }
                for idx, row in df.iterrows()
                if idx != selected_idx
            ]
            nearest = sorted(distances, key=lambda x: x["Distance"])[:5]

            figure = {
                "data": [
                    {
                        "x": [features[x_feature] for features in df["PCA_Features"]],
                        "y": [features[y_feature] for features in df["PCA_Features"]],
                        "text": df["Podcast"],
                        "mode": "markers",
                        "marker": {"size": 12, "color": "blue"},
                    },
                    {
                        "x": [selected_features[x_feature]],
                        "y": [selected_features[y_feature]],
                        "text": ["Selected Podcast"],
                        "mode": "markers",
                        "marker": {"size": 15, "color": "red", "symbol": "star"},
                    },
                ],
                "layout": {
                    "title": f"PCA Visualization (Feature {x_feature+1} vs Feature {y_feature+1})",
                    "xaxis": {"title": f"PCA Feature {x_feature+1}"},
                    "yaxis": {"title": f"PCA Feature {y_feature+1}"},
                },
            }

            # Search nearest podcast on Spotify
            nearest_podcast = nearest[0]["Podcast"] if nearest else None
            token = get_spotify_token()
            if token and nearest_podcast:
                try:
                    search_timeout = 5  # Set a time limit (seconds)
                    start_time = time.time()
                    response = None
                    while time.time() - start_time < search_timeout:
                        url = "https://api.spotify.com/v1/search"
                        headers = {"Authorization": f"Bearer {token}"}
                        params = {"q": nearest_podcast, "type": "show", "limit": 1}
                        response = requests.get(url, headers=headers, params=params)
                        if response.status_code == 200:
                            break

                    if response and response.status_code == 200:
                        spotify_result = response.json()["shows"]["items"][0]
                        print(f"Nearest Podcast Found on Spotify: {spotify_result['name']} - {spotify_result['external_urls']['spotify']}")
                    else:
                        print("Could not fetch nearest podcast details within time limit.")

                except Exception as e:
                    print(f"Error fetching nearest podcast from Spotify: {e}")

            return figure, nearest

        except Exception as e:
            print(f"Error during visualization update: {e}")
            return {}, []
