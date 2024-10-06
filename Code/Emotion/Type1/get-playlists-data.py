import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import util
import pandas as pd
import math
import pickle

features_for_mood = ['energy', 'liveness', 'tempo', 'speechiness',
                                     'acousticness', 'instrumentalness', 'danceability', 'duration_ms',
                                     'loudness', 'valence']

def get_track_features(track_ids, spotify):
    chunk_size = 50
    num_chunks = int(math.ceil(len(track_ids) / float(chunk_size)))
    features_add = []
    for i in range(num_chunks):
        chunk_track_ids = track_ids[i * chunk_size:min((i + 1) * chunk_size, len(track_ids))]
        chunk_features = spotify.audio_features(tracks=chunk_track_ids)
        features_add.extend(chunk_features)

    features_df = pd.DataFrame(features_add).drop(['id', 'analysis_url', 'key', 'mode', 'time_signature',
                                                   'track_href', 'type', 'uri'], axis=1)
    features_df = features_df[features_for_mood]
    return features_df

token = util.prompt_for_user_token(username='username', scope='user-library-read',
                                                   client_id='clientid',
                                                   client_secret='clientsecret',
                                                   redirect_uri='http://localhost:5000')
spotify = spotipy.Spotify(auth=token, requests_timeout=20)

playlists = {
             'Energetic' : ["https://open.spotify.com/playlist/4QDWboU5rwpDRXwYprwJf5?si=815ef93592e7475f",
                             "https://open.spotify.com/playlist/0V32mTwWBzo6rNIk21owsY?si=1421ae7c97434fbf"]
             'Relaxing':    ["https://open.spotify.com/playlist/1r4hnyOWexSvylLokn2hUa?si=a622dcb83906450f",
                             "https://open.spotify.com/playlist/11IcIUefRdjIpy1K5GMdOH?si=83b430249c854434",]
             'Dark':        ["https://open.spotify.com/playlist/52fjgY5XTdBKrk71TJks1i?si=28df39510a12411e",
                             "https://open.spotify.com/playlist/5xLGNk2Hqi47S1skLPIIZg?si=bb90318b09644f6a",]
            'Aggressive':  ["https://open.spotify.com/playlist/0y1bZzglw6D3t5lPXRmuYS?si=57c0db5e01b9420b",
                             "https://open.spotify.com/playlist/7rthAUYUFcbEeC8NS8Wh42?si=ab7f353f5f2847b3",]
            'Happy' :      ["https://open.spotify.com/playlist/1h90L3LP8kAJ7KGjCV2Xfd?si=5e2691af69e544a3",
                             "https://open.spotify.com/playlist/4AnAUkQNrLKlJCInZGSXRO?si=9846e647548d4026",]
}

tracks = pd.DataFrame()
moods = []

for mood, links in playlists.items():
    print (mood)
    for link in links:
        id = link[34:56]
        try:
            pl_tracks = spotify.playlist_tracks(id)['items']
            ids = [foo['track']['id'] for foo in pl_tracks]
        except:
            print (link)
            continue
        features = get_track_features(ids, spotify)
        features['id'] = ids
        features['mood'] = mood
        tracks = tracks.append(features)

tracks.to_csv('tracks2.csv')