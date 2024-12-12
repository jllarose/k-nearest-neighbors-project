# Handle imports up-front
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

movies=pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv")
credits=pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_credits.csv")

# Combine the datasets
merged_ds = pd.merge(movies, credits, on = "title")

merged_ds = merged_ds[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Make a copy to work with while encoding so that we have the original to go back to
# if needed
encoded_data_df=merged_ds.copy()

# Empty list to hold extracted values
extracted_values=[]

# Loop on the elements of the cast column
for json_string in merged_ds['cast']:

    # Load the json string into a python dictionary
    json_list=json.loads(json_string)

    # Empty list to hold values from this element
    values=[]

    # Loop on the first three elements of the json list
    for item in json_list[:3]:

        # Extract the value for the name key
        value=item['name']

        # Add it to the list
        values.append(value)

    extracted_values.append(values)

# Replace the 'cast' column with the extracted values
encoded_data_df['cast']=extracted_values

# Same for the 'keywords' column
encoded_data_df['keywords']=merged_ds['keywords'].apply(lambda x: [item['name'] for item in json.loads(x)][:3] if pd.notna(x) else None)

# And the 'genres' column
encoded_data_df['genres']=merged_ds['genres'].apply(lambda x: [item['name'] for item in json.loads(x)][:3] if pd.notna(x) else None)


# And the 'crew' column so that it is just the director
encoded_data_df["crew"] = encoded_data_df["crew"].apply(lambda x: " ".join([crew_member['name'] for crew_member in json.loads(x) if crew_member['job'] == 'Director']))

# Converting the Overview column to a list
encoded_data_df['overview']=merged_ds['overview'].apply(lambda x: [x if pd.notna(x) else 'none'])

# Creating the 'tags' column
encoded_data_df["tags"]=encoded_data_df["overview"] + encoded_data_df["genres"] + encoded_data_df["keywords"] + encoded_data_df["cast"]

# Converting 'tag' column to one long string instead of list
encoded_data_df['tags'] = encoded_data_df['tags'].apply(lambda x: ', '.join(x))

tags = encoded_data_df['tags']

# K Nearest Neighbor Model Training:
vectorize = TfidfVectorizer()
tfidf_matrix=vectorize.fit_transform(tags)

model = NearestNeighbors(n_neighbors = 5, algorithm = 'brute', metric = 'cosine')
model.fit(tfidf_matrix)

def get_movie_recommendations(movie_title):
    movie_index = encoded_data_df[encoded_data_df["title"] == movie_title].index[0]
    distances, indices = model.kneighbors(tfidf_matrix[movie_index])
    similar_movies = [(encoded_data_df["title"][i], distances[0][j]) for j, i in enumerate(indices[0])]
    return similar_movies[1:]

input_movie = "How to Train Your Dragon"
recommendations = get_movie_recommendations(input_movie)
print("Film recommendations '{}'".format(input_movie))
for movie, distance in recommendations:
    print("- Film: {}".format(movie))