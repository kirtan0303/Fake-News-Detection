from flask import Flask, render_template, request, jsonify
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Load models and vectorizer
with open('models/naive_bayes.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('models/random_forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Define preprocessing function
def preprocess_text(text):
    """Clean and preprocess text"""
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra spaces
    text = re.sub('\s+', ' ', text).strip()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back into text
    text = ' '.join(tokens)
    
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input text
        text = request.form['text']
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Vectorize text
        text_vector = vectorizer.transform([processed_text])
        
        # Make predictions
        nb_prediction = nb_model.predict(text_vector)[0]
        nb_probability = round(nb_model.predict_proba(text_vector)[0][nb_prediction] * 100, 2)
        
        rf_prediction = rf_model.predict(text_vector)[0]
        rf_probability = round(rf_model.predict_proba(text_vector)[0][rf_prediction] * 100, 2)
        
        # Format results
        nb_result = "Real" if nb_prediction == 1 else "Fake"
        rf_result = "Real" if rf_prediction == 1 else "Fake"
        
        # Combine results (simple ensemble)
        combined_prediction = "Real" if (nb_prediction + rf_prediction) / 2 > 0.5 else "Fake"
        combined_confidence = (nb_probability + rf_probability) / 2
        
        # Return results
        results = {
            'text': text,
            'naive_bayes': {
                'prediction': nb_result,
                'confidence': nb_probability
            },
            'random_forest': {
                'prediction': rf_result,
                'confidence': rf_probability
            },
            'combined': {
                'prediction': combined_prediction,
                'confidence': combined_confidence
            }
        }
        
        return render_template('result.html', results=results)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if request.method == 'POST':
        # Get input from JSON
        data = request.json
        text = data.get('text', '')
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Vectorize text
        text_vector = vectorizer.transform([processed_text])
        
        # Make predictions
        nb_prediction = nb_model.predict(text_vector)[0]
        nb_probability = float(nb_model.predict_proba(text_vector)[0][nb_prediction])
        
        rf_prediction = rf_model.predict(text_vector)[0]
        rf_probability = float(rf_model.predict_proba(text_vector)[0][rf_prediction])
        
        # Format results
        nb_result = "Real" if nb_prediction == 1 else "Fake"
        rf_result = "Real" if rf_prediction == 1 else "Fake"
        
        # Combine results (simple ensemble)
        combined_prediction = "Real" if (nb_prediction + rf_prediction) / 2 > 0.5 else "Fake"
        combined_confidence = (nb_probability + rf_probability) / 2
        
        # Return results
        results = {
            'text': text,
            'naive_bayes': {
                'prediction': nb_result,
                'confidence': round(nb_probability * 100, 2)
            },
            'random_forest': {
                'prediction': rf_result,
                'confidence': round(rf_probability * 100, 2)
            },
            'combined': {
                'prediction': combined_prediction,
                'confidence': round(combined_confidence * 100, 2)
            }
        }
        
        return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
