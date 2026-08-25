FROM python:3.11-slim

WORKDIR /app

# Install dependencies first, in their own layer — copying only the
# app's own light requirements.txt (not the heavy project-root one used
# for training/DVC/CI) means `docker build` only reinstalls packages
# when requirements change, not on every app code edit.
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual app code
COPY app/ /app/

# Only the preprocessor is needed locally — the trained model is
# loaded at runtime from the MLflow/DagsHub registry
# (models:/URLGuard_model@staging), not from a local pickle. There is
# no vectorizer.pkl in this project (feature engineering here is a
# ColumnTransformer, not TF-IDF), so nothing else is copied from models/.
COPY models/preprocessor.pkl /app/models/preprocessor.pkl

# No NLTK download step — this project's feature extraction is
# URL-lexical + live HTML-content parsing (requests/BeautifulSoup),
# not NLP text classification, so there's no stopwords/wordnet corpus
# to fetch.

EXPOSE 8000

# local — single process, auto-detects code changes if bind-mounted
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# prod — multiple worker processes via gunicorn, each running FastAPI
# through uvicorn's ASGI worker class (gunicorn itself is WSGI-only,
# this is the standard way to run an ASGI app under it)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "120", "main:app"]