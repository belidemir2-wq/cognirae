# Streamlit App

## Local run
1. Install deps: `pip install -r requirements.txt`
2. Run: `streamlit run app.py`

## Deploy on Streamlit Community Cloud
1. Push this folder (app.py and requirements.txt at root) to GitHub.
2. In Streamlit Community Cloud, choose Deploy from GitHub, select the repo/branch, and set main file path to `app.py`.
3. Add your secret under Secrets: `OPENAI_API_KEY = "your-key"`.

## Troubleshooting
- Missing package: add to requirements.txt, commit, push; redeploy.
- Missing key: set OPENAI_API_KEY in Secrets.
- Wrong entry point: ensure app.py is at repo root.
