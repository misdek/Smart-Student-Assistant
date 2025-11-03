import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipediaapi
import os


class PlagiarismDetector:
    def __init__(self, dataset_csv='dataset/plagiarism_combined.csv'):
        self.dataset_csv = dataset_csv

        # Check dataset existence
        if not os.path.exists(dataset_csv):
            raise FileNotFoundError(f"Dataset not found: {dataset_csv}")

        # Load dataset
        self.df = pd.read_csv(dataset_csv)

        # TF-IDF vectorizer on local dataset
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                          stop_words='english',
                                          max_features=20000)
        docs = self.df['content'].fillna('').tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(docs)

        # Wikipedia API client (with user-agent required)
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='SmartStudentAssistant/1.0 (contact: youremail@example.com)'
        )

    # ------------------------------
    # Local dataset similarity
    # ------------------------------
    def check_tfidf(self, text: str, top_k: int = 5):
        q_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]

        results = []
        for i in top_idx:
            results.append({
                'id': int(self.df.iloc[i]['id']),
                'title': self.df.iloc[i]['title'],
                'score': float(sims[i]),
                'content': str(self.df.iloc[i]['content'])[:300]
            })
        return results

    # ------------------------------
    # Wikipedia snippet fetching
    # ------------------------------
    def fetch_wikipedia_snippets(self, query: str, max_results: int = 5):
        try:
            titles = self.wiki.search(query, results=max_results)
        except Exception:
            titles = []

        snippets = []
        for t in titles:
            try:
                p = self.wiki.page(t)
                if p and p.exists():
                    txt = p.summary if hasattr(p, 'summary') and p.summary else p.text[:1000]
                    snippets.append({'title': p.title, 'content': txt})
            except Exception:
                continue
        return snippets

    # ------------------------------
    # Wikipedia similarity checking
    # ------------------------------
    def check_with_wikipedia(self, text: str, top_k: int = 5):
        snippets = self.fetch_wikipedia_snippets(text, max_results=top_k * 2)
        if not snippets:
            return []

        docs = [s['content'] for s in snippets]
        vec = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
        vec_docs = vec.fit_transform(docs)
        q_vec = vec.transform([text])

        sims = cosine_similarity(q_vec, vec_docs).flatten()
        top_idx = sims.argsort()[::-1][:top_k]

        results = []
        for i in top_idx:
            results.append({
                'title': snippets[i]['title'],
                'score': float(sims[i]),
                'content': snippets[i]['content'][:500]
            })
        return results

    # ------------------------------
    # Combined plagiarism checker
    # ------------------------------
    def check_plagiarism(self, text: str, top_k: int = 5):
        if not text.strip():
            return {"error": "No text provided"}

        # --- Local TF-IDF check ---
        local_results = self.check_tfidf(text, top_k=top_k)
        # --- Wikipedia check ---
        wiki_results = self.check_with_wikipedia(text, top_k=top_k)

        # Convert similarity (0–1) → percentage (0–100)
        for r in local_results:
            r["score"] = round(float(r["score"]) * 100, 2)
        for r in wiki_results:
            r["score"] = round(float(r["score"]) * 100, 2)

        # Use consistent keys that match the HTML file
        return {
            "local_matches": local_results,
            "wikipedia_matches": wiki_results
        }

