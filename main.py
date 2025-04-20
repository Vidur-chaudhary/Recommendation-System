from fastapi import FastAPI # type: ignore
from pydantic import BaseModel
from model import recommend_assessments  # <- Import your logic
from fastapi import Response
import pandas as pd
import os
import json
from fastapi.middleware.cors import CORSMiddleware
from evaluate import compute_recall_at_k, compute_map_at_k
import requests
from io import StringIO
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluationRequest(BaseModel):
    query: str
    relevant_keywords: list[str]
    k: int = 5

@app.post("/evaluate")
def evaluate(input: EvaluationRequest):
    results = recommend_assessments(input.query)
    recall = compute_recall_at_k(results, input.relevant_keywords, input.k)
    mapk = compute_map_at_k(results, input.relevant_keywords, input.k)

    return {
        "query": input.query,
        "recall_at_k": round(recall, 2),
        "map_at_k": round(mapk, 2),
        "recommendations": results[:input.k]
    }

#show csv data in json form 


@app.get("/assessments")
def list_assessments(limit: int = 10):
    # GitHub raw CSV URL
    csv_url = "https://raw.githubusercontent.com/Vidur-chaudhary/Recommendation-System/main/shl_assessments.csv"

    try:
        # Fetch the CSV file content from GitHub
        response = requests.get(csv_url)
        
        # Check if the request was successful
        if response.status_code != 200:
            return Response(
                content=json.dumps({"error": f"Failed to retrieve CSV. Status code: {response.status_code}"}, indent=2),
                media_type="application/json"
            )

        # Use StringIO to read the CSV content as a file-like object
        df = pd.read_csv(StringIO(response.text))
        preview = df.head(limit)
        json_data = preview.to_dict(orient="records")

        # Return JSON response
        return Response(
            content=json.dumps(json_data, indent=2),
            media_type="application/json"
        )

    except Exception as e:
        # Handle any exceptions that occur
        return Response(
            content=json.dumps({"error": str(e)}, indent=2),
            media_type="application/json"
        )


class QueryInput(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(input: QueryInput):
    return recommend_assessments(input.query)
