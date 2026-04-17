import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge on id and movie_id
movies = movies.merge(credits, left_on='id', right_on='movie_id')
movies = movies.drop(['movie_id', 'title_y'], axis=1)
movies = movies.rename(columns={'title_x': 'title'})

# Function to extract names from JSON
def extract_names(json_str, key='name', top=3):
    try:
        data = ast.literal_eval(json_str)
        names = [item[key] for item in data[:top]]
        return ' '.join(names)
    except:
        return ''

# Extract genres, keywords, cast
movies['genres'] = movies['genres'].apply(lambda x: extract_names(x))
movies['keywords'] = movies['keywords'].apply(lambda x: extract_names(x))
movies['cast'] = movies['cast'].apply(lambda x: extract_names(x))

# Combine into tags
movies['tags'] = movies['overview'].fillna('') + ' ' + movies['genres'] + ' ' + movies['keywords'] + ' ' + movies['cast']

# Lowercase and remove punctuation
movies['tags'] = movies['tags'].str.lower()
movies['tags'] = movies['tags'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Vectorize
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend
def recommend(title):
    idx = movies[movies['title'] == title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Test
print(recommend("Columbia Pictures"))