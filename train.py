import pandas as pd
import numpy as np
import re
import string
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

def load_data():
    """Load and combine fake and real news datasets"""
    # For this example, we're using the Kaggle "Fake and Real News" dataset
    # You would need to download this dataset first
    # https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
    
    # Load datasets
    fake = pd.read_csv('data/Fake.csv')
    real = pd.read_csv('data/True.csv')
    
    # Add labels
    fake['label'] = 0  # 0 for fake
    real['label'] = 1  # 1 for real
    
    # Combine datasets
    df = pd.concat([fake, real], axis=0, ignore_index=True)
    
    # Keep only text and label columns
    df = df[['text', 'label']]
    
    return df

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

def prepare_data(df):
    """Prepare data for training"""
    # Preprocess text
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42
    )
    
    # TF-IDF Vectorization
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

def train_models(X_train, y_train):
    """Train Naive Bayes and Random Forest models"""
    # Train Naive Bayes
    print("Training Naive Bayes model...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    
    # Train Random Forest
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return nb_model, rf_model

def evaluate_models(models, X_test, y_test):
    """Evaluate trained models"""
    model_names = ['Naive Bayes', 'Random Forest']
    results = {}
    
    for i, model in enumerate(models):
        name = model_names[i]
        print(f"Evaluating {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix
        }
        
        # Print results
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} Classification Report:")
        print(report)
        print(f"{name} Confusion Matrix:")
        print(conf_matrix)
        print("-" * 50)
    
    return results

def save_models(models, vectorizer):
    """Save trained models and vectorizer"""
    model_names = ['naive_bayes', 'random_forest']
    
    # Create models directory if it doesn't exist
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save models
    for i, model in enumerate(models):
        with open(f'models/{model_names[i]}.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    # Save vectorizer
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Models and vectorizer saved successfully!")

def visualize_results(results):
    """Visualize model performance"""
    # Set up the figure
    plt.figure(figsize=(12, 10))
    
    # Plot confusion matrices
    plt.subplot(2, 1, 1)
    sns.heatmap(results['Naive Bayes']['confusion_matrix'], annot=True, 
                fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.title('Naive Bayes Confusion Matrix')
    
    plt.subplot(2, 1, 2)
    sns.heatmap(results['Random Forest']['confusion_matrix'], annot=True, 
                fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.title('Random Forest Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    
    # Bar chart of accuracies
    plt.figure(figsize=(8, 6))
    accuracies = [results['Naive Bayes']['accuracy'], results['Random Forest']['accuracy']]
    plt.bar(['Naive Bayes', 'Random Forest'], accuracies, color=['skyblue', 'lightgreen'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    
    print("Visualizations saved successfully!")

def main():
    """Main function to run the training pipeline"""
    print("Loading data...")
    df = load_data()
    
    print("Data overview:")
    print(f"Total samples: {len(df)}")
    print(f"Fake news samples: {len(df[df['label'] == 0])}")
    print(f"Real news samples: {len(df[df['label'] == 1])}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(df)
    
    # Train models
    nb_model, rf_model = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models([nb_model, rf_model], X_test, y_test)
    
    # Save models
    save_models([nb_model, rf_model], vectorizer)
    
    # Visualize results
    visualize_results(results)
    
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
