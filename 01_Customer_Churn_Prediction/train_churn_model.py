import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_churn_prediction_pipeline(data_path='churn_data.csv'):
    """
    A beginner-friendly Machine Learning pipeline to predict Customer Churn.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, data_path)
    
    print("====================================")
    print("1. LOADING THE DATA")
    print("====================================")
    
    # 1. Load the dataset into a pandas DataFrame
    if not os.path.exists(file_path):
        print(f"Error: {data_path} not found. Please run dataset_generator.py first!")
        return
        
    df = pd.read_csv(file_path)
    print(df.head())
    print("\nData Shape (Rows, Columns):", df.shape)
    
    # Check what percentage of our customers actually churned
    churn_rate = df['Churn'].mean() * 100
    print(f"\nOverall Churn Rate in Dataset: {churn_rate:.1f}%")

    print("\n====================================")
    print("2. DATA PREPROCESSING (CLEANING)")
    print("====================================")
    
    # Machine Learning models can only read numbers (not text like "Month-to-month")
    # We use LabelEncoder to turn 'Contract_Type' text into numbers (0, 1, 2)
    le = LabelEncoder()
    df['Contract_Type_Code'] = le.fit_transform(df['Contract_Type'])
    
    # Let's print the mapping so we know what number means what contract
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"Converted Contract Types to numbers: {mapping}")

    # Dropping columns we don't need for the model
    # CustomerID is just an ID, it doesn't help predict behavior
    # Contract_Type is text which we just converted to 'Contract_Type_Code'
    # Churn is what we want to predict (the Target), so it can't be an input feature
    X = df.drop(['CustomerID', 'Contract_Type', 'Churn'], axis=1) # The input features
    y = df['Churn'] # The answer (Target)

    print("\nFeatures used for training:", X.columns.tolist())

    print("\n====================================")
    print("3. SPLITTING THE DATASET")
    print("====================================")
    
    # We split our data: 80% to train the model, 20% to test how smart it got
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data size: {X_train.shape[0]} customers")
    print(f"Testing data size: {X_test.shape[0]} customers")

    print("\n====================================")
    print("4. TRAINING THE MACHINE LEARNING MODEL")
    print("====================================")
    
    # We create a Random Forest Model (a powerful algorithm that builds many decision trees)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("Training the Random Forest on the 80% data...")
    # .fit() tells the model to learn the patterns between X_train and y_train
    model.fit(X_train, y_train)
    print("Training Complete!")

    print("\n====================================")
    print("5. EVALUATING THE MODEL")
    print("====================================")
    
    # Now we ask our model to predict the answers for the 20% test data it has never seen
    predictions = model.predict(X_test)
    
    # We compare the model's predictions to the actual answers (y_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Detailed Classification Report:")
    # The classification report shows Precision, Recall, and F1-score 
    # (very important metrics for an ML internship interview!)
    print(classification_report(y_test, predictions, target_names=["Stayed (0)", "Churned (1)"]))

    print("\n====================================")
    print("6. FEATURE IMPORTANCE (WHY DID IT PREDICT CHURN?)")
    print("====================================")
    
    # Random Forests can tell us which features were most important when making decisions!
    importances = model.feature_importances_
    features = X.columns
    
    # Zip them together and sort them so the most important is at the top
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    print("Top factors driving customer churn:")
    print(feature_importance_df)
    
    # Let's visualize this with a bar chart, employers love visualizations!
    try:
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, hue='Feature', palette='viridis', legend=False)
        plt.title('What Drives Customer Churn? (Feature Importance)')
        plt.xlabel('Importance Score')
        plt.ylabel('Customer Feature')
        plt.tight_layout()
        
        # Save the plot as an image in the project folder
        plot_path = os.path.join(script_dir, 'feature_importance.png')
        plt.savefig(plot_path)
        print(f"\n=> Created a visualization and saved it as '{plot_path}'")
        plt.close() # Close the plot figure
    except Exception as e:
        print("Could not generate plot chart. Make sure matplotlib and seaborn are installed.")
        print(f"Error: {e}")

if __name__ == "__main__":
    run_churn_prediction_pipeline()
