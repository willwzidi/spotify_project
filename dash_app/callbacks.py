from dash import Input, Output, State, html
from spotify_helpers import get_spotify_token, search_podcasts, generate_clustering_data
from scipy.spatial.distance import euclidean
import pandas as pd

def register_callbacks(app):
    @app.callback(
        [
            Output("search-results", "children"),
            Output("podcast-dropdown", "options"),
            Output("clustering-data-store", "data"),  # Store clustering data
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
        [Input("podcast-dropdown", "value")],
        [State("clustering-data-store", "data")],  # Access clustering data from store
    )
    def update_visualization(selected_idx, clustering_data):
        if selected_idx is None or clustering_data is None:
            return {}, []

        try:
            df = pd.DataFrame(clustering_data)
            selected_metrics = df.iloc[selected_idx][["Metric 1", "Metric 2", "Metric 3"]]
            features = df[["Metric 1", "Metric 2", "Metric 3"]].values

            distances = [
                {
                    "Podcast": row["Podcast"],
                    "Episode": row["Episode"],
                    "Distance": euclidean(features[selected_idx], features[idx]),
                }
                for idx, row in df.iterrows() if idx != selected_idx
            ]
            nearest = sorted(distances, key=lambda x: x["Distance"])[:5]

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
