import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# Preprocessing function (ensure it matches the training script)
def preprocess_data(df, label_encoders=None):
    # Drop 'Unnamed: 0' if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Convert 'trans_date_trans_time' to datetime if it exists
    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%d-%m-%Y %H:%M')
        df['year'] = df['trans_date_trans_time'].dt.year
        reference_date = pd.to_datetime('2020-01-01')
        df['time_since_reference'] = (df['trans_date_trans_time'] - reference_date).dt.days
        df = df.drop('trans_date_trans_time', axis=1)
    
    # Encode non-numeric columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    if label_encoders:
        for col in non_numeric_cols:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            else:
                raise ValueError(f"Label encoder for column '{col}' not found.")
    
    return df

# Load the trained model
model = joblib.load('trained_model.pkl')
print("Model loaded successfully.")

# Load the saved label encoders
label_encoders = joblib.load('label_encoders.pkl')

# Load test data (replace 'path_to_test_data.csv' with actual path or test data)
X_test = pd.read_csv('fraudTrain.csv')

# Ensure target column 'is_fraud' is not in test data
if 'is_fraud' in X_test.columns:
    X_test = X_test.drop('is_fraud', axis=1)

# Preprocess the test data
X_test = preprocess_data(X_test, label_encoders)

# Predict with the model
predictions = model.predict(X_test)
print("Sample predictions:", predictions[:5])
