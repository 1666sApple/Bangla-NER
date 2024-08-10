from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.utils.predict import predict_sentence

router = APIRouter()

class SentenceInput(BaseModel):
    sentence: str

@router.post("/predict")
async def predict(sentence_input: SentenceInput):
    try:
        result = predict_sentence(sentence_input.sentence)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
