# Logistic Regression From Scratch

A from-scratch implementation of simple and multiple logistic regression using NumPy, Pandas, and Matplotlib without relying on scikit-learn or similar libraries.

## Setup

Create and activate a virtual environment:

| Windows | Linux/macOS |
|---|---|
| `python -m venv venv`<br>`venv\Scripts\activate` | `python3 -m venv .venv`<br>`source .venv/bin/activate` |

Install dependencies:
```bash
pip install -r requirements.txt
```

If you add a new dependency, update `requirements.txt` before pushing:
```bash
pip freeze > requirements.txt
```

## Usage

If `dataset/Social_Network_Ads.csv` is not present yet, run:
```bash
python load_dataset.py
```

Why this step matters:
- Jupyter notebooks expect a local copy of `dataset/Social_Network_Ads.csv`
- `load_dataset.py` downloads the Kaggle dataset once, moves the CSV into the expected folder, and prints a quick preview so you can confirm the file loaded correctly
- If the CSV already exists locally, the script detects that and skips the download