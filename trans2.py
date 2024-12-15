import pandas as pd

# Load the CSV file into a DataFrame
file_path = "C:/Users/thiru/Downloads/archive/imdb_top_1000.csv"  # Replace with your file's path if it's in a different directory
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame

# Dropping the extra columns from the data frame MOVIES which will be used for training the model as well as creating embeddings
df = df[['Genre','Series_Title','IMDB_Rating','Overview']]
MOVIES = df
# Apply the custom function to each row using the apply()
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Function to get BERT embeddings for a given text
def get_bert_embedding(text):
    # Encode the input text and create attention mask
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors="pt",
        return_attention_mask=True
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)

    embeddings = output.last_hidden_state[:, 0, :].squeeze().numpy()  # Squeezing to get 1D embeddings
    return embeddings

# Calculate movie embeddings "movie_ef"
movie_ef = np.array([get_bert_embedding(desc) for desc in MOVIES['Overview']])
print(movie_ef) #movie_ef contains embeddings of MOVIES['tags'] as a numpy array

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Function to get BERT embeddings for a given text
def get_bert_embedding(text):
    # Encode the input text and create attention mask
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors="pt",
        return_attention_mask=True
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)

    embeddings = output.last_hidden_state[:, 0, :].numpy()  # Squeezing to get 1D embeddings
    return embeddings

def preprocess_user_input(user_input):
    # Preprocess the input for better tokenization
    user_input = user_input.lower()
    user_input = re.sub(r'[^\w\s]', ' ', user_input)  # Remove punctuation
    user_input = re.sub(r'\s+', ' ', user_input)  # Normalize whitespace
    return user_input

# Extract user-specified movie names within double inverted commas
def extract_movie_names(user_input):
    movie_names = re.findall(r'"([^"]*)"', user_input)
    return movie_names

# Extract user-specified imdb rating mentioned after imdb:
def extract_imdb_rating(user_input):
    
    imdb_rating = re.findall(r'imdb:(\d+\.\d+)', user_input)       #imdb:8.2
    if len(imdb_rating) == 0:
         imdb_rating = re.findall(r'imdb: (\d+\.\d+)', user_input) #imdb: 8.2
    if len(imdb_rating) == 0 :
         imdb_rating = re.findall(r'imdb:(\d+)', user_input)       #imdb:8
    if len(imdb_rating) == 0 :
        imdb_rating = re.findall(r'imdb: (\d+)', user_input)       #imdb: 8
    return imdb_rating

# Function to get movie embeddings based on user-specified movie names
def get_movie_embeddings(movie_names):
    movie_embeddings = []
    
    for name in movie_names:
        matching_movie = MOVIES[MOVIES['Series_Title'] == name]
        if not matching_movie.empty:
            movie_index = MOVIES[MOVIES['Series_Title'] == name].index[0]
            movie_embedding = movie_ef[movie_index]
            movie_embeddings.append(movie_embedding)
    return np.array(movie_embeddings)


# Calculate similarity using weighted cosine similarity
def calculate_weighted_similarity(user_embedding, user_movie_embeddings, movie_embeddings):
    user_similarities = cosine_similarity(user_embedding.reshape(1, -1), movie_embeddings) #reshaping the user_embeddings to 2D
    if not user_movie_embeddings.size == 0: 
        movie_similarities = cosine_similarity(user_movie_embeddings, movie_embeddings)
        weighted_similarities = 0.5 * user_similarities + 0.5 * movie_similarities
    else: 
        weighted_similarities = 1 * user_similarities
    return weighted_similarities

# Function to recommend top 15 movies based on user input
def recommend_movies(user_input):
    user_input = user_input 
    user_inputs = preprocess_user_input(user_input)
    
    # Tokenize the user input and get attention mask
    inputs = tokenizer(user_inputs, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Use the model to get embeddings for the user input
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)

    user_embedding = output.last_hidden_state[:, 0, :].numpy()  
 

    movies = MOVIES.copy()

    # Filter the movies based on IMDb score
    imdb_score = extract_imdb_rating(user_input) 
    for x in imdb_score:
        imdb_score = float(x) 
    if imdb_score:
            movies = MOVIES[MOVIES['IMDB_Rating'] >= imdb_score]

    filtered_indices = movies.index
    filtered_embeddings = movie_ef[filtered_indices]
    
    # Calculate similarity using weighted cosine similarity
    user_movie_names = extract_movie_names(user_input)
    user_movie_embeddings = get_movie_embeddings(user_movie_names)
    weighted_similarity_scores = calculate_weighted_similarity(user_embedding, user_movie_embeddings, filtered_embeddings)
    
    # Get the indices of the top 15 similar movies
    movie_indices = np.argsort(weighted_similarity_scores[0])[-15:]

    # Get the recommended movies based on the indices
    recommended_movies = movies.iloc[movie_indices]['Series_Title'].tolist()
    return recommended_movies

# Example usage
if __name__ == "__main__":
    print(" 1. If you are mentioning any movie than mention the full name of the movie between double inverted commas, example: \"Mission: Impossible\"")
    print(" 2. Metion the IMDB rating more than or equal to which you want your movie to be as imdb:8")
    user_input = input("Enter your prompt: ")
    print()
    print("Prompt: ",user_input)
    recommended_movies = recommend_movies(user_input)
    print("\nRecommended Movies:")
    for i, movie in enumerate(recommended_movies, 1):
        print(f"{i}. {movie}")


