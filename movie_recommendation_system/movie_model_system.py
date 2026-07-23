import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dataset = pd.read_csv('movies.csv')

selected_keywords = ["genres", "keywords", "tagline", "cast", "director"]
for feature in selected_keywords:
    dataset[feature] = dataset[feature].fillna("")

combined_features = (
    dataset["genres"] + " " + dataset["keywords"] + " " +
    dataset["tagline"] + " " + dataset["cast"] + " " + dataset["director"]
)

vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vector)

list_of_all_titles = dataset["title"].tolist()

def recommend(movie_name, top_n=19):
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    if not find_close_match:
        print(f"No close match found for '{movie_name}'.")
        return

    close_match = find_close_match[0]
    print(f"Closest match: {close_match}\n")

    index_of_the_movie = dataset[dataset.title == close_match].index[0]

    similar_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_score = sorted(similar_score, key=lambda x: x[1], reverse=True)

    print("Movies suggested for you:\n")
    for i, (index, score) in enumerate(sorted_similar_score[1:top_n+1], start=1):
        title_from_index = dataset.iloc[index]["title"]
        print(i, '.', title_from_index)

while True:
    movie_name = input("Enter the movie name (or 'q' to quit): ")
    if movie_name.lower() == 'q':
        break
    recommend(movie_name)
    print()