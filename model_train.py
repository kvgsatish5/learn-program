import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Create a dummy dataset for job predictions
# features: [stream, percentage, age, skills_count]
# streams: 0: CS, 1: ME, 2: EE, 3: Commerce
def create_dataset():
    np.random.seed(42)
    n_samples = 1000
    
    streams = np.random.randint(0, 4, n_samples)
    percentages = np.random.randint(50, 100, n_samples)
    ages = np.random.randint(18, 30, n_samples)
    skills_count = np.random.randint(1, 10, n_samples)
    
    # Simple logic for target 'job_type'
    # 0: Software Engineer, 1: Data Analyst, 2: Mechanical Engineer, 3: Business Analyst
    y = []
    for i in range(n_samples):
        if streams[i] == 0 and percentages[i] > 75:
            y.append(0) # Software
        elif streams[i] == 3 and percentages[i] > 70:
            y.append(3) # Business
        elif streams[i] == 1:
            y.append(2) # Mechanical
        else:
            y.append(1) # Data Analyst (default)
            
    df = pd.DataFrame({
        'stream': streams,
        'percentage': percentages,
        'age': ages,
        'skills_count': skills_count,
        'job_type': y
    })
    return df

# 2. Train the model
def train():
    print("Generating dataset...")
    df = create_dataset()
    
    X = df.drop('job_type', axis=1)
    y = df['job_type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model trained. Accuracy: {accuracy * 100:.2f}%")
    
    # 3. Save the model
    joblib.dump(model, 'model.pkl')
    print("Model saved as model.pkl")

if __name__ == "__main__":
    train()
