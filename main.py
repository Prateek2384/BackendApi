from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model/gbc_lead_scorer.pkl")

class LeadData(BaseModel):
    Credit_Score: int
    Age_Group: str
    Family_Background: str
    Income: int
    Comments: str

@app.post("/score")
def score_lead(data: LeadData):
    try:
        input_df = pd.DataFrame([data.dict()])
        proba = model.predict_proba(input_df)[0][1]
        initial_score = round(proba * 100)

        reranked_score = rerank_score(initial_score, data.Comments)

        return {
            "initial_score": initial_score,
            "reranked_score": max(0, min(100, reranked_score)),
            "comments": data.Comments
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def rerank_score(score, comment):
    comment = comment.lower()
    adjustments = {
        "urgent": 10,
        "asap": 10,
        "need help": 8,
        "last option": 5,
        "not interested": -10,
        "just browsing": -8,
        "can we talk soon": 3
    }

    for word, val in adjustments.items():
        if word in comment:
            score += val

    return max(0, min(100, score))