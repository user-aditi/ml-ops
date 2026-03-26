# Iris Classification Model 🌸

## 📌 Project Objective

This project demonstrates a simple machine learning pipeline to classify iris flower species using a Logistic Regression model. It covers data loading, preprocessing, model training, evaluation, and saving the trained model.

---

## 📊 Dataset

* Dataset used: Iris dataset
* Source: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
* Features:

  * sepal_length
  * sepal_width
  * petal_length
  * petal_width
* Target:

  * species (setosa, versicolor, virginica)

---

## ⚙️ Project Structure

```
project/
│
├── data/                # (Optional) for datasets
├── src/
│   └── train.py         # Training script
├── models/
│   └── iris_model.pkl   # Saved model
├── requirements.txt
└── README.md
```

---

## 🚀 Steps to Run

### 1. Create Virtual Environment (optional but recommended)

```bash
python -m venv venv
```

Activate it:

* Windows:

```bash
venv\Scripts\activate
```

* Mac/Linux:

```bash
source venv/bin/activate
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run Training Script

```bash
python src/train.py
```

---

## 📈 Output

* Prints model accuracy in the console
* Saves trained model in:

```
models/iris_model.pkl
```

---

## 🧠 Model Used

* Logistic Regression (from scikit-learn)

---

## 📦 Dependencies

* pandas
* scikit-learn
* joblib

---

## ✅ Summary

This project is a beginner-friendly example of:

* Data loading
* Model training
* Evaluation
* Model persistence

You can extend this project by:

* Adding visualization
* Using different models
* Deploying the model as an API

---

Happy Learning! 🚀
