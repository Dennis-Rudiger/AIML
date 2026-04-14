# Project 1: Customer Churn Prediction 📉

This is a complete Machine Learning pipeline for predicting customer churn. It takes raw customer data and uses a **Random Forest Classifier** to learn the patterns of customers who leave.

## Skills Demonstrated
*   **Data Generation & Processing:** `pandas`, `numpy`, Handling Categorical Variables (`LabelEncoder`)
*   **Machine Learning Modeling:** `scikit-learn`, `RandomForestClassifier`
*   **Model Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrices
*   **Data Visualization:** `matplotlib`, `seaborn`

## 📁 Files Explained

1.  **`dataset_generator.py`**: This script automatically builds a synthetic dataset (`churn_data.csv`). I added specific internal logic (e.g., customers who make more support calls or are on month-to-month contracts are mathematically more likely to churn).
2.  **`train_churn_model.py`**: This is the core machine learning pipeline. It loads the CSV, cleans the data, splits it into training/testing sets (80/20), trains a Random Forest model, and spits out a detailed Classification Report and a Feature Importance graph.

## 🚀 How to Run (Beginner Friendly)

Make sure you've installed the requirements from the main folder:
```powershell
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Step 1:** Generate the data by running:
```powershell
python dataset_generator.py
```
*(This will create a file called `churn_data.csv` in your folder)*

**Step 2:** Train the Machine Learning Model:
```powershell
python train_churn_model.py
```

## What the Model Will Tell You 💡
The script prints exactly what happens during training. Most importantly, it outputs a **Classification Report** and saves a picture (`feature_importance.png`) revealing the top reasons *why* customers are canceling their subscriptions.
