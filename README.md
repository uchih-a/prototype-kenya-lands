# Kenya Land Valuation — Streamlit App

MLP-powered land price estimator for Kenya, deployed on Streamlit Community Cloud.

## Project Structure

```
kenya_land_valuation_app/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
└── models/                 ← ⚠️ Add your model files here
    ├── mlp_model.pt
    ├── mlp_scaler.pkl
    └── mlp_feature_list.pkl
```

## Step 1 — Add Your Model Files

Copy from your Google Drive outputs folder:
- `mlp_model.pt` — PyTorch model state dict (from Cell M10)
- `mlp_scaler.pkl` — Fitted StandardScaler
- `mlp_feature_list.pkl` — Ordered feature names list

Place all three inside a `models/` folder at the project root.

## Step 2 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Kenya Land Valuation Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/kenya-land-valuation.git
git push -u origin main
```

## Step 3 — Deploy on Streamlit Community Cloud (FREE)

1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repo, branch (`main`), and set `app.py` as the main file
5. Click **"Deploy!"**

Your app will be live at:
`https://YOUR_USERNAME-kenya-land-valuation.streamlit.app`

## Step 4 — Update the App

Any `git push` to your repo automatically redeploys the app — no manual steps needed.

## Notes

- The `models/` folder with `.pt` and `.pkl` files **must** be committed to GitHub for the app to work.
- If your model files are large (> 100 MB), use [Git LFS](https://git-lfs.github.com/).
- Streamlit Community Cloud gives **1 free app** with always-on hosting.
