import pandas as pd
import numpy as np
import os

def create_synthetic_churn_data(num_samples=1000, output_file='churn_data.csv'):
    """
    Generates a beginner-friendly synthetic dataset for Customer Churn Prediction.
    
    What is Churn? 
    Churn means a customer has stopped using our service (e.g., cancelled their subscription).
    """
    print(f"Generating {num_samples} customer records...")
    
    # Set a random seed so the results are exactly the same every time we run this code
    np.random.seed(42)
    
    # 1. Generate Customer Features (The information we know about the customer)
    # Tenure: How many months the customer has been with us (1 to 72 months)
    tenure = np.random.randint(1, 73, num_samples)
    
    # MonthlyCharges: How much they pay per month ($20 to $120)
    monthly_charges = np.random.uniform(20.0, 120.0, num_samples)
    
    # SupportCalls: How many times they called customer support recently (0 to 10)
    support_calls = np.random.randint(0, 11, num_samples)
    
    # ContractType: The type of contract they are on (Month-to-month, One year, Two year)
    contracts = ['Month-to-month', 'One year', 'Two year']
    contract_type = np.random.choice(contracts, num_samples, p=[0.5, 0.3, 0.2])
    
    # 2. Simulate the 'Churn' Logic (Target Variable)
    # We create a hidden probability of them leaving based on their features.
    # Higher support calls -> more likely to churn
    # Month-to-month -> more likely to churn
    # High tenure -> less likely to churn
    churn_prob = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Base probability
        prob = 0.1
        
        # If they call support a lot, they might be angry/frustrated
        if support_calls[i] > 3:
            prob += 0.3
            
        # Month-to-month customers can leave easily
        if contract_type[i] == 'Month-to-month':
            prob += 0.3
            
        # Loyal customers are less likely to leave
        if tenure[i] > 24:
            prob -= 0.2
            
        # Ensure probability stays between 0 and 1
        prob = max(0.0, min(1.0, prob))
        churn_prob[i] = prob
        
    # Generate the actual Churn (1 = Yes, 0 = No) based on the hidden probability
    # If a random number is less than their churn probability, they are marked as Churned
    churn = (np.random.random(num_samples) < churn_prob).astype(int)
    
    # 3. Combine everything into a pandas DataFrame
    df = pd.DataFrame({
        'CustomerID': range(1001, 1001 + num_samples),
        'Tenure_Months': tenure,
        'Monthly_Charges': np.round(monthly_charges, 2),
        'Support_Calls': support_calls,
        'Contract_Type': contract_type,
        'Churn': churn
    })
    
    # 4. Save the dataset to a CSV file so our machine learning model can read it later
    df.to_csv(output_file, index=False)
    print(f"Dataset successfully created and saved as '{output_file}'!")
    print(f"Total customers: {len(df)}")
    print(df.head()) # Preview the first 5 rows

if __name__ == "__main__":
    # Ensure we run this from the correct folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'churn_data.csv')
    
    create_synthetic_churn_data(output_file=output_path)
