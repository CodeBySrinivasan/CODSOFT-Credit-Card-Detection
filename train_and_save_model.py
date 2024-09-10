import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Preprocessing function
def preprocess_data(df, label_encoders=None, is_training=True):
    # Drop 'Unnamed: 0' if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Convert 'trans_date_trans_time' to datetime if it exists
    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%d-%m-%Y %H:%M')
        # Extract year from the datetime column
        df['year'] = df['trans_date_trans_time'].dt.year
        # Calculate days since a reference date (example: '2020-01-01')
        reference_date = pd.to_datetime('2020-01-01')
        df['time_since_reference'] = (df['trans_date_trans_time'] - reference_date).dt.days
        df = df.drop('trans_date_trans_time', axis=1)  # Drop original datetime column
    
    # Encode non-numeric columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    if label_encoders is None and is_training:
        label_encoders = {}
        for col in non_numeric_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        # Use the existing label encoders for the test set
        for col in non_numeric_cols:
            df[col] = label_encoders[col].transform(df[col].astype(str))

    return df, label_encoders

# Load your data
df = pd.read_csv('fraudTrain.csv')  # Replace with your actual data file path

# Preprocess the data (this will return the processed DataFrame and label encoders)
df, label_encoders = preprocess_data(df)

# Save label encoders for future use
joblib.dump(label_encoders, 'label_encoders.pkl')

# Define the target column
target_column = 'is_fraud'

# Check if target column exists in DataFrame
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in the DataFrame. Available columns: {df.columns}")

# Split the data into features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the LogisticRegression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Print classification report to evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'trained_model.pkl')

print("Model training complete and saved.")
