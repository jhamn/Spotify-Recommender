import spotipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white')
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

cid = '<INSERT YOUR CLIENT ID HERE>'  # Client ID; copy this from your app
secret = '<INSERT CLIENT SECRET HERE>'  # Client Secret; copy this from your app
username = '<INSERT YOUR USERNAME HERE>'  # Your Spotify username
CACHE = '.cache-' + username
# for avaliable scopes see https://developer.spotify.com/web-api/using-scopes/
scope = 'user-library-read playlist-modify-public playlist-read-private'

redirect_uri = 'http://localhost:8080'  # Paste your Redirect URI here

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

token = util.prompt_for_user_token(username=username, scope=scope, client_id=cid, client_secret=secret, redirect_uri=redirect_uri, cache_path=CACHE)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)

sourcePlaylistID = '<INSERT PLAYLIST-ID>'
sourcePlaylist = sp.user_playlist(username, sourcePlaylistID);
tracks = sourcePlaylist["tracks"];
songs = tracks["items"];

track_ids = []
track_names = []

for i in range(0, len(songs)):
    if songs[i]['track']['id'] != None:  # Removes the local tracks in your playlist if there is any
        track_ids.append(songs[i]['track']['id'])
        track_names.append(songs[i]['track']['name'])

features = []
for i in range(0, len(track_ids)):
    audio_features = sp.audio_features(track_ids[i])
    for track in audio_features:
        features.append(track)

playlist_df = pd.DataFrame(features, index=track_names)

playlist_df = playlist_df[["id", "acousticness", "danceability", "duration_ms", "energy", "instrumentalness",  "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]]

print(playlist_df)

playlist_df['ratings'] = [9, 7, 7, 8, 9, 7, 6, 5, 6, 5, 8, 8, 8, 9]

X_train = playlist_df.drop(['id', 'ratings'], axis=1)
y_train = playlist_df['ratings']

X_scaled = StandardScaler().fit_transform(X_train)

pca = decomposition.PCA().fit(X_scaled)

plt.figure(figsize=(10, 7))
plt.plot(pd.np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 12)
plt.yticks(pd.np.arange(0, 1.1, 0.1))
plt.axvline(7, c='b') # Tune this so that you obtain at least a 95% total variance explained
plt.axhline(0.95, c='r')
plt.show()

# Fit your dataset to the optimal pca
pca = decomposition.PCA(n_components=6)
X_pca = pca.fit_transform(X_scaled)

v = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 6), max_features=10000)
X_names_sparse = v.fit_transform(track_names)
print(X_names_sparse.shape)

X_train_last = csr_matrix(hstack([X_pca, X_names_sparse]))

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

knn_params = {'n_neighbors': range(1, 10)}
knn = KNeighborsClassifier(n_jobs=-1)
knn_grid = GridSearchCV(knn, knn_params, cv=skf, n_jobs=-1, verbose=True)
knn_grid.fit(X_train_last, y_train)
print(knn_grid.best_params_, knn_grid.best_score_)

parameters = {'max_features': [4, 7, 8, 10], 'min_samples_leaf': [1, 3, 5, 8], 'max_depth': [3, 5, 8]}
rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
forest_grid = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
forest_grid.fit(X_train_last, y_train)
print(forest_grid.best_estimator_, forest_grid.best_score_)

tree = DecisionTreeClassifier()
tree_params = {'max_depth': range(1, 11), 'max_features': range(2, 19)}
tree_grid = GridSearchCV(tree, tree_params, cv=skf, n_jobs=-1, verbose=True)
tree_grid.fit(X_train_last, y_train)
print(tree_grid.best_estimator_, tree_grid.best_score_)

rec_tracks = []
for i in playlist_df['id'].values.tolist():
    rec_tracks += sp.recommendations(seed_tracks=[i], limit=int(len(playlist_df) / 2))['tracks'];

rec_track_ids = []
rec_track_names = []
for i in rec_tracks:
    rec_track_ids.append(i['id'])
    rec_track_names.append(i['name'])

rec_features = []
for i in range(0, len(rec_track_ids)):
    rec_audio_features = sp.audio_features(rec_track_ids[i])
    for track in rec_audio_features:
        rec_features.append(track)

rec_playlist_df = pd.DataFrame(rec_features, index=rec_track_ids)

rec_playlist_df = rec_playlist_df[["acousticness", "danceability", "duration_ms", "energy", "instrumentalness",  "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]]

tree_grid.best_estimator_.fit(X_train_last, y_train)
rec_playlist_df_scaled = StandardScaler().fit_transform(rec_playlist_df)
X_test_pca = pca.transform(rec_playlist_df_scaled)
X_test_names = v.transform(rec_track_names)
X_test_last = csr_matrix(hstack([X_test_pca, X_test_names]))
y_pred_class = tree_grid.best_estimator_.predict(X_test_last)

rec_playlist_df['ratings']=y_pred_class
rec_playlist_df = rec_playlist_df.sort_values('ratings', ascending = False)
rec_playlist_df = rec_playlist_df.reset_index()

# Pick the top ranking tracks to add your new playlist; 9 or 10 will work
recs_to_add = rec_playlist_df[rec_playlist_df['ratings']>=9]['index'].values.tolist()

playlist_recs = sp.user_playlist_create(username, name='PCA + tf-idf + DT - Recommended Songs for Playlist - {}'.format(sourcePlaylist['name']))

sp.user_playlist_add_tracks(username, playlist_recs['id'], recs_to_add)
