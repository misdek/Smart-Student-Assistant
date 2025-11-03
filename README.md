# Smart Student Assistant (NLP) - Full Project

This package contains a FastAPI-based project that provides:
- Text summarization (TextRank by default, optional transformers)
- Plagiarism detection against a local 2000-entry dataset and live Wikipedia snippets
- Simple web UI to submit text and view results

## How plagiarism with Wikipedia works
The app will, at runtime, query the Wikipedia API (via `wikipedia-api`) using the input text as a search query, fetch top article summaries and check similarity between the input and those summaries. This provides a lightweight live-check against Wikipedia.

## Run locally
1. Create a virtualenv and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   uvicorn app:app --reload --port 8000
   ```
3. Open http://localhost:8000 in your browser.

## Notes
- Live Wikipedia checking requires internet access from where the app runs.
- For large-scale production, precompute vector indices and use ANN (Faiss/Annoy) for faster search.
