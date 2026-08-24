# 🚢 Titanic Survival Prediction — Live Web App

This project deploys a Machine Learning model as an interactive Streamlit web application.

## 📌 Task

**Deploy Your Model as a Live Web App**

The app takes passenger information as input and predicts whether the passenger is likely to survive.

## 🧠 Model

- Algorithm: Logistic Regression
- Preprocessing: Missing-value imputation, StandardScaler and OneHotEncoder
- Dataset: Titanic dataset provided for this task
- Test Accuracy: **80.45%**
- Model format: `model.joblib`

## 📂 Project Structure

```text
TechNova_Task_Deploy_Model_Live_App/
│
├── app.py
├── model.joblib
├── train_model.py
├── requirements.txt
├── README.md
│
└── data/
    └── titanic.csv
```

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Live App

After deploying on Streamlit Community Cloud, replace the placeholder below with your live URL:

**Live Demo:** `PASTE_YOUR_STREAMLIT_APP_LINK_HERE`

## 🚀 Streamlit Community Cloud Deployment

1. Upload this complete folder to a GitHub repository.
2. Make sure `app.py`, `model.joblib`, and `requirements.txt` are in the repository root.
3. Open Streamlit Community Cloud.
4. Connect your GitHub account.
5. Select this repository.
6. Select `app.py` as the main file.
7. Click **Deploy**.
8. Copy the generated live URL.
9. Replace `PASTE_YOUR_STREAMLIT_APP_LINK_HERE` in this README with your live URL.

## 📝 Files

- `app.py` — Streamlit web application
- `model.joblib` — trained machine learning pipeline
- `train_model.py` — reproducible model training script
- `data/titanic.csv` — dataset
- `requirements.txt` — required Python packages

## 👨‍💻 Technologies

Python • Pandas • Scikit-learn • Joblib • Streamlit
