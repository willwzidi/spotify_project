import requests
import pandas as pd
import numpy as np
import json
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer

# Spotify API credentials
CLIENT_ID = 'fc668f7d3ed94d9f9d0f4f8e9c8af9ad'
CLIENT_SECRET = 'feb9efe5c0224e92a44c52e22aa5b9a2'

# Load PCA model and vocabulary
PCA_MODEL = load("incremental_pca_model.pkl")
with open("vocabulary.json", "r") as file:
    VOCABULARY = json.load(file)

VECTORIZE = CountVectorizer(vocabulary=VOCABULARY)


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


def search_podcasts(query, token, limit = 50):
    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": "show", "limit": 10}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        results = response.json()["shows"]["items"]
        return results
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []


def generate_clustering_data(podcasts):
    data = []
    for podcast in podcasts:
        data.append({
            "Podcast": podcast["name"],
            "Episode": podcast["description"][:50] + "...",
            "Metric 1": np.random.uniform(0.5, 1.0),
            "Metric 2": np.random.uniform(0.5, 1.0),
            "Metric 3": np.random.uniform(0.5, 1.0),
        })
    return pd.DataFrame(data)


def use_pca(data):
    text_data = data["Podcast"].tolist()
    text_features = VECTORIZE.transform(text_data)
    pca_features = PCA_MODEL.transform(text_features.toarray())
    return pca_features