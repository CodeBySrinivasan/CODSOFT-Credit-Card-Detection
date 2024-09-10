from flask import Flask, request, jsonify, render_template
import joblib
import traceback
from preprocess_data import preprocess_data  # Import the function

# Initialize the Flask app
app = Flask(__name__)

# Load the model and label encoders
def load_model_and_scaler():
    try:
        # Load the trained model
        model = joblib.load('trained_model.pkl')  # Ensure this path is correct
        
        # Load the label encoders
        label_encoders = joblib.load('label_encoders.pkl')  # Ensure this path is correct
        return model, label_encoders
    except Exception as e:
        print(f"Error loading model or label encoders: {str(e)}")
        raise

# Define route to render the index.html file
@app.route('/')
def index():
    return render_template('index.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoders is None:
        return jsonify({'error': 'Model or encoders are not loaded correctly'})
    
    try:
        # Get form data from the POST request
        input_data = request.form.to_dict()

        # Check if input data is not empty
        if not input_data:
            return jsonify({'Error': '☠FRAUD TRANSACTION☠'})

        # Preprocess the input data
        processed_data = preprocess_data(input_data, label_encoders)
        
        # Make predictions
        prediction = model.predict(processed_data)
        
        # Return the prediction result as JSON
        return jsonify({'prediction': int(prediction[0])})  # Convert to int to avoid NumPy issues
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

# Load the model and label encoders when starting the Flask app
if __name__ == '__main__':
    try:
        model, label_encoders = load_model_and_scaler()
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Failed to start the app: {str(e)}")
