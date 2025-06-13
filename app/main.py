from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag_pipeline import answer_question

app = FastAPI(title="ChatING API")


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(payload: QuestionRequest):
    try:
        result = answer_question(payload.question)
        print(f"Result: ", result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
