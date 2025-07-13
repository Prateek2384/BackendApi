from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr, validator
from typing import Optional
import joblib
import pandas as pd
import re

app = FastAPI()

# Load model
try:
    model = joblib.load("model/gbc_lead_scorer.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    raise Exception(f"Model load failed: {e}")

# Store scored leads in memory
leads_db = []

class LeadData(BaseModel):
    Credit_Score: int
    Age_Group: str
    Family_Background: str
    Income: int
    Lead_Source: str
    Product_Interest_Level: str
    Interaction_Frequency: str
    Comments: Optional[str] = ""
    
    # Validation
    @validator('Credit_Score')
    def valid_credit_score(cls, v):
        if not (300 <= v <= 850):
            raise ValueError('Credit score must be between 300 and 850')
        return v
    
    @validator('Income')
    def valid_income(cls, v):
        if v < 0:
            raise ValueError('Income must be non-negative')
        return v

@app.post("/score")
def score_lead(data: LeadData):
    try:
        input_df = pd.DataFrame([{
            "Credit Score": data.Credit_Score,
            "Age Group": data.Age_Group,
            "Family Background": data.Family_Background,
            "Income": data.Income,
            "Lead Source": data.Lead_Source,
            "Product Interest Level": data.Product_Interest_Level,
            "Interaction Frequency": data.Interaction_Frequency
        }])
        
        proba = model.predict_proba(input_df)[0][1]
        initial_score = round(proba * 100)

        reranked_score = rerank_score(initial_score, data.Comments)

        lead_entry = {
            "credit_score": data.Credit_Score,
            "income": data.Income,
            "lead_source": data.Lead_Source,
            "product_interest": data.Product_Interest_Level,
            "initial_score": initial_score,
            "reranked_score": max(0, min(100, reranked_score)),
            "comments": data.Comments
        }

        leads_db.append(lead_entry)

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
        "high priority": 8,
        "important": 6,
        "not interested": -10,
        "no thanks": -10,
        "just browsing": -8,
        "can we talk soon": 5
    }

    for word, val in adjustments.items():
        if word in comment:
            score += val

    return max(0, min(100, score))

@app.get("/leads")
def get_leads():
    return {"leads": leads_db}