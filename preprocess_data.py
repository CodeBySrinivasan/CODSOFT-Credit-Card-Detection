import pandas as pd

def preprocess_data(input_data, label_encoders):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Ensure columns match expected feature names
        expected_features = ['category', 'amount', 'city']  # Example list; adjust as needed
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = ''  # Set to default or empty value based on feature type

        # Reorder columns to match model's expected order
        df = df[expected_features]

        # Encode categorical columns with previously fitted label encoders
        for col in label_encoders:
            if col in df.columns:
                df[col] = label_encoders[col].transform(df[col].astype(str))

        return df
    except Exception as e:
        print(f"Error during data preprocessing: {str(e)}")
        raise
