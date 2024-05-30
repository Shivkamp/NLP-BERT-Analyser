# Segment 1: Importing Libraries and Reading Data

import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizer, BertModel
import numpy as np  # Add this import statement for NumPy

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Rest of your code follows...


df = pd.read_csv("tmdb_5000_movies.csv")
df.head(1)


# Segment 2: Preprocessing Genres and Keywords
def genres_and_keywords_to_string(row):
    genres = row["genres"]
    if pd.isna(genres):
        genres = ""
    else:
        genres = json.loads(genres)
        genres = " ".join("".join(j["name"].split()) for j in genres)

    keywords = row["keywords"]
    if pd.isna(keywords):
        keywords = ""
    else:
        keywords = json.loads(keywords)
        keywords = " ".join("".join(j["name"].split()) for j in keywords)

    return "%s %s" % (genres, keywords)


df["string"] = df.apply(genres_and_keywords_to_string, axis=1)

# Define maximum sequence length
MAX_SEQ_LENGTH = 128

# Segment 3: Generate BERT Embeddings


def generate_bert_embeddings(row):
    if pd.isnull(row["overview"]):  # Check for missing values in the 'overview' column
        return None  # Return None if the 'overview' value is missing

    text = row["overview"]  # Get the overview column value
    # Tokenize input
    tokens = tokenizer.tokenize(text)
    # Truncate or pad the tokens to the specified max length
    tokens = tokens[: MAX_SEQ_LENGTH - 2]  # -2 for [CLS] and [SEP]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids += [0] * (MAX_SEQ_LENGTH - len(input_ids))  # Pad tokens to the max length
    input_ids = torch.tensor([input_ids])

    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs[0][
            :, 0, :
        ].numpy()  # Extract embeddings of the first (and only) token [CLS]

    return embeddings


# Apply the function to each row of the DataFrame
df["bert_embeddings"] = df.apply(generate_bert_embeddings, axis=1)

# Segment 4: User Input and Querying

while True:
    try:
        input_movie = str(input("Enter movie title: "))
        idx = df[df["title"] == input_movie].index[0]
        break
    except IndexError:
        print("Wrong movie title. Please enter a valid movie title.")

query_embedding = df.at[idx, "bert_embeddings"]

# Segment 5: Calculate Cosine Similarity and Plot Scores


def calculate_cosine_similarity(query_embedding, embeddings):
    scores = []
    for embedding in embeddings:
        if embedding is not None:
            score = cosine_similarity([query_embedding], [embedding.reshape(-1)])
            scores.append(score[0][0])
        else:
            scores.append(np.nan)  # Handle missing embeddings
    return scores


scores = calculate_cosine_similarity(query_embedding.reshape(-1), df["bert_embeddings"])

print("Scores:", scores)  # Debugging statement to check the scores

plt.plot(scores)
plt.show()  # Ensure plot is displayed

# Segment 6: Recommend Top Titles

# Convert scores to a NumPy array
scores_array = np.array(scores)

# Get the indices of top recommendations
recommended_idx = (-scores_array).argsort()[1:6]

# Get the titles of the recommended movies
recommended_titles = df["title"].iloc[recommended_idx]

# Print the top 5 recommended titles
print("Top 5 Recommended Titles:")
print("--------------------------")
for i, title in enumerate(recommended_titles, 1):
    print(f"{i}. {title}")
