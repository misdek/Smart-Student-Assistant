from typing import Optional
USE_TRANSFORMERS = False
try:
    if USE_TRANSFORMERS:
        from transformers import pipeline
        summarizer_pipe = pipeline('summarization')
    else:
        summarizer_pipe = None
except Exception:
    summarizer_pipe = None

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

def summarize_text(text: str, max_sentences: int = 3) -> str:
    if summarizer_pipe is not None:
        try:
            res = summarizer_pipe(text, max_length=130, min_length=30, do_sample=False)
            return res[0]['summary_text']
        except Exception:
            pass
    parser = PlaintextParser.from_string(text, Tokenizer('english'))
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, max_sentences)
    return ' '.join([str(s) for s in summary_sentences])
