from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from app.utils.predict import predict_sentence

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

class PredictionInput(BaseModel):
  sentence: str

class PredictionOutput(BaseModel):
  prediction: List[List[str]]  # List of lists for token, pos, ner tags

@app.get("/")
async def index(request: Request):
  return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionOutput)
async def predict(input: PredictionInput):
  result = predict_sentence(input.sentence)
  return PredictionOutput(prediction=result)


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
