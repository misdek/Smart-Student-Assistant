from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from summarizer import summarize_text
from plagiarism import PlagiarismDetector

app = FastAPI(title='Smart Student Assistant')

app.mount('/static', StaticFiles(directory='static'), name='static')

# Initialize plagiarism detector (will build TF-IDF index)
PDET = PlagiarismDetector(dataset_csv='dataset/plagiarism_combined.csv')

class SummRequest(BaseModel):
    text: str
    max_sentences: int = 3

class PlagRequest(BaseModel):
    text: str
    method: str = 'tfidf'  # or 'lsh' (if implemented)
    top_k: int = 5



@app.post('/summarize')
async def summarize(req: SummRequest):
    short = summarize_text(req.text, max_sentences=min(2, req.max_sentences))
    long = summarize_text(req.text, max_sentences=req.max_sentences)
    return JSONResponse({'short_summary': short, 'long_summary': long})

@app.post('/plagiarism')
async def plagiarism(req: PlagRequest):
    # checks both local dataset and live Wikipedia snippets
    local_matches = PDET.check_tfidf(req.text, top_k=req.top_k)
    wiki_matches = PDET.check_with_wikipedia(req.text, top_k=req.top_k)

    # convert scores to percentage
    for r in local_matches:
        r["score"] = round(float(r["score"]) * 100, 2)
    for r in wiki_matches:
        r["score"] = round(float(r["score"]) * 100, 2)

    # overall plagiarism = max of all scores
    all_scores = [r["score"] for r in local_matches + wiki_matches]
    overall_score = max(all_scores) if all_scores else 0

    return JSONResponse({
        'overall_score': overall_score,
        'local_matches': local_matches,
        'wikipedia_matches': wiki_matches
    })


@app.get('/')
async def root():
    with open('static/index.html', 'r', encoding='utf8') as f:
        return HTMLResponse(f.read())

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
