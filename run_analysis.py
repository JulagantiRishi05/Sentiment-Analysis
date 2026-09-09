import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add local path
sys.path.append(os.path.abspath('.'))
from prepare_dataset import generate_sentiment_dataset
from src.preprocess import TextPreprocessor

def run_pipeline():
    os.makedirs('data', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    
    # 1. Dataset Generation & Loading
    if not os.path.exists('data/sentiment_dataset.csv'):
        generate_sentiment_dataset()
    else:
        generate_sentiment_dataset() # ensure fresh generation
        
    df = pd.read_csv('data/sentiment_dataset.csv')
    print("Dataset Loaded. Total samples:", len(df))
    print(df['sentiment'].value_counts())
    
    # 2. Visualization: Class Distribution
    plt.figure(figsize=(10, 5))
    palette = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}
    
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='sentiment', palette=palette, order=['positive', 'negative', 'neutral'])
    plt.title('Sentiment Class Counts', fontsize=12, fontweight='bold')
    plt.xlabel('Sentiment Class')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    counts = df['sentiment'].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=[palette[k] for k in counts.index], startangle=140, explode=(0.03, 0.03, 0.03))
    plt.title('Sentiment Class Proportion', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('images/class_distribution.png', dpi=300)
    plt.close()
    print("Saved images/class_distribution.png")

    # 3. Preprocessing
    preprocessor = TextPreprocessor()
    df['clean_text'] = preprocessor.preprocess_series(df['text'])
    
    # 4. WordClouds for each class
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, s_class in enumerate(['positive', 'negative', 'neutral']):
        class_text = " ".join(df[df['sentiment'] == s_class]['clean_text'])
        wc = WordCloud(width=600, height=400, background_color='white', colormap='viridis' if s_class=='positive' else ('Reds' if s_class=='negative' else 'Blues')).generate(class_text)
        
        # Save individual wordclouds as well
        wc.to_file(f'images/wordcloud_{s_class}.png')
        
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].set_title(f'WordCloud: {s_class.capitalize()} Sentiment', fontsize=14, fontweight='bold')
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('images/wordcloud_combined.png', dpi=300)
    plt.close()
    print("Saved WordClouds to images/")

    # 5. Train / Test Split
    X = df['clean_text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, df.index, test_size=0.20, random_state=42, stratify=y
    )
    
    # 6. TF-IDF Feature Extraction
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 7. Model Training & Evaluation
    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(random_state=42)
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        predictions[name] = y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        results[name] = {
            'Accuracy': acc,
            'Precision (Macro)': prec,
            'Recall (Macro)': rec,
            'F1-Score (Macro)': f1,
            'Report': classification_report(y_test, y_pred)
        }
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1-Score: {f1:.4f}")
        print(results[name]['Report'])

    # Plot Confusion Matrices Side by Side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ['negative', 'neutral', 'positive']
    
    for i, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[i])
        axes[i].set_title(f'Confusion Matrix: {name}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
        
    plt.tight_layout()
    plt.savefig('images/confusion_matrices.png', dpi=300)
    plt.close()
    print("Saved images/confusion_matrices.png")

    # 8. Error Analysis (using Logistic Regression or Naive Bayes)
    test_df = df.loc[test_indices].copy()
    test_df['predicted_nb'] = predictions['Multinomial Naive Bayes']
    test_df['predicted_lr'] = predictions['Logistic Regression']
    
    misclassified = test_df[test_df['sentiment'] != test_df['predicted_lr']].head(5)
    print("\n=== Error Analysis (Top Misclassifications) ===")
    for idx, row in misclassified.iterrows():
        print(f"Original Text: '{row['text']}'")
        print(f"Clean Text:    '{row['clean_text']}'")
        print(f"True Label:    {row['sentiment']}")
        print(f"Predicted LR:  {row['predicted_lr']}")
        print(f"Predicted NB:  {row['predicted_nb']}")
        print("-" * 50)
        
    print("\nPipeline Execution Finished Successfully!")

if __name__ == '__main__':
    run_pipeline()
