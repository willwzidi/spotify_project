import requests
import pandas as pd
import numpy as np

# Spotify API credentials
CLIENT_ID = 'fc668f7d3ed94d9f9d0f4f8e9c8af9ad'
CLIENT_SECRET = 'feb9efe5c0224e92a44c52e22aa5b9a2'

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
    data = []
    for podcast in podcasts:
        data.append({
            "Podcast": podcast["name"],
            "Episode": podcast["description"][:50] + "...",
            "Metric 1": podcast.get("popularity", np.random.uniform(0.5, 1.0)),
            "Metric 2": podcast.get("total_episodes", np.random.randint(5, 100)) / 100,
            "Metric 3": np.random.uniform(0.5, 1.0),
        })
    return pd.DataFrame(data)
