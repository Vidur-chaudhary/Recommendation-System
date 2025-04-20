import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
genai.configure(api_key="API_KEY")

# Load the DataFrame from .pkl file
with open("shl_assessments_with_embeddings.pkl", "rb") as f:
    df = pickle.load(f)

# Convert the 'embedding' column into a matrix
embeddings = np.vstack(df["embedding"].values)  # shape: (n, 768)

# Gemini embedding function (yours should already be working)
def get_embedding(text: str):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return response['embedding']

# Final recommendation function
def recommend_assessments(query: str, top_k: int = 10):
    query_vector = np.array(get_embedding(query)).reshape(1, -1)
    similarities = cosine_similarity(query_vector, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            "assessment_name": row["name"],
            "url": row["url"],
            "remote_testing": row["remote_support"],      # corrected key
            "adaptive_support": row["adaptive_support"],
            "duration": "Not provided",                    # optional: add if you later extract it
            "test_type": row["test_type"],
            "similarity_score": float(similarities[idx])   # optional: useful for debugging
        })

    return results

# Example usage
if __name__ == "__main__":
    query = "Looking for a sales manager with good communication and leadership"
    recommendations = recommend_assessments(query)
    for rec in recommendations:
        print(rec)
