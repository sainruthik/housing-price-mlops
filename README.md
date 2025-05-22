# 🏠 Housing Price Prediction App (MLOps + Streamlit + Docker)

This project predicts the sale price of residential homes using the Ames Housing dataset. It’s built as an end-to-end machine learning pipeline using:

- ✅ Streamlit for the frontend
- ✅ XGBoost & Linear Regression models
- ✅ MLflow for experiment tracking
- ✅ Docker for reproducible deployment
- ✅ Render for cloud hosting

---

## 🚀 Live Demo

👉 [Click here to try the app](https://housing-price-app.onrender.com)  
_(replace with your actual Render or HuggingFace URL)_

---

## 🔧 Features

- Predict house price based on user input
- Compare models: **XGBoost** vs **Linear Regression**
- Trained on Ames Housing dataset
- One-hot encoded features + scaling
- CI/CD with auto-deploy to Render on GitHub push
- Dockerized for portable deployment

---

## 🗂️ Project Structure

```
.
├── dashboard/           # Streamlit app
├── models/              # Saved models (.pkl)
├── src/                 # Data prep, prediction logic
├── data/                # Input dataset (ames.csv)
├── Dockerfile           # For containerization
├── requirements.txt     # Python dependencies
```

---

## 🧠 Model Performance

| Model            | RMSE     | R² Score |
|------------------|----------|----------|
| XGBoost          | 23660.5  | 0.93     | 
| Linear Regression| 28880.6  | 0.896    |

_(values may vary slightly depending on preprocessing)_

---

## 🐳 Run Locally with Docker

```bash
# Build Docker image
docker build -t housing-price-app .

# Run container
docker run -p 8501:8501 housing-price-app
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Dataset Source

- [Ames Housing Dataset – Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)

---

## 👨‍💻 Author

**Sai Nruthik Sri Harsha Kuruva**  
Deployed using [Render.com](https://render.com)  
Built with ❤️ using Python, Streamlit, and Docker
