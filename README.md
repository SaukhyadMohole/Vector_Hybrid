## Streamlit Retrieval UI

This app provides a UI to run Dense, Sparse (BM25), and Hybrid Fusion retrieval, preview results, and display evaluation metrics at @5.

### Files
- `streamlit_app.py` — Streamlit UI
- `retrieval_backend.py` — adapter that imports your existing retrieval functions or falls back to mocks; also includes metrics
- `requirements.txt` — Python dependencies
- `runtime.txt` — Python version for Streamlit Cloud
- `sample.py` — your existing code (if it defines retrieval functions)

### Local run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Deploy (Streamlit Community Cloud)
1. Push this folder to a GitHub repo.
2. On https://share.streamlit.io, create a new app:
   - Repository: `<your-username>/<your-repo>`
   - Branch: `master` or `main`
   - Main file path: `streamlit_app.py`
3. If you have secrets, add them in the app's settings.


