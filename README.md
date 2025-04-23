# Fake-News-Detection

A machine learning-powered web application that helps identify potential fake news articles using NLP techniques and ensemble classification models.

## Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning to detect fake news from real news articles.
It implements both Naive Bayes and Random Forest classifiers, combining their predictions for better accuracy.

### Key Features

- Text preprocessing pipeline (tokenization, stemming, stopword removal)
- TF-IDF vectorization for feature extraction
- Ensemble of Naive Bayes and Random Forest models
- Interactive web interface with Flask
- Detailed prediction results with confidence scores
- REST API endpoint for programmatic access

## Technology Stack

- **Python**: Core programming language
- **Scikit-learn**: Machine learning models and preprocessing
- **NLTK**: Natural language processing tools
- **Flask**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Bootstrap**: Front-end styling
- **HTML/CSS**: User interface
  

## Model Training and Evaluation

The model training pipeline includes:

1. Data preprocessing: Cleaning, tokenization, stopword removal, and stemming
2. Feature extraction: TF-IDF vectorization with 5000 features
3. Model training: Naive Bayes and Random Forest
4. Model evaluation: Accuracy, precision, recall, F1-score, and confusion matrix


## API Documentation

The application provides a REST API endpoint for programmatic use

### POST /api/predict

Analyzes text for fake news detection.

**Request:**
```json
{
  "text": "Your news article text here..."
}
```

**Response:**
```json
{
  "text": "Your news article text here...",
  "naive_bayes": {
    "prediction": "Fake",
    "confidence": 95.67
  },
  "random_forest": {
    "prediction": "Fake",
    "confidence": 92.34
  },
  "combined": {
    "prediction": "Fake",
    "confidence": 94.01
  }
}
```

## Future Improvements

- Implement more advanced NLP techniques (word embeddings, transformers)
- Add more models to the ensemble (LSTM, BERT)
- Improve UI/UX with interactive visualizations
- Add user authentication and history tracking
- Deploy as a browser extension

## License

This project is licensed under the MIT License
