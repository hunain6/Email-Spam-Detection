import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('punkt')

from google.colab import files
print("Please upload your dataset (CSV file with 'text' and 'label' columns)")
print("Or we'll create a sample dataset for you")

try:
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    df = pd.read_csv(filename)
    print(f"\nDataset loaded successfully! Shape: {df.shape}")
except:
    print("\nNo file uploaded. Creating sample dataset...")
    np.random.seed(42)
    n_samples = 10000
    
    spam_words = ['free', 'win', 'prize', 'cash', 'click', 'offer', 'urgent', 'congratulations', 'winner', 'lottery', 'money', 'credit', 'loan', 'viagra', 'discount', 'buy now', 'limited', 'guaranteed']
    ham_words = ['meeting', 'project', 'report', 'schedule', 'dinner', 'movie', 'weekend', 'thanks', 'please', 'attached', 'update', 'review', 'discussion', 'lunch', 'breakfast']
    
    texts = []
    labels = []
    
    for i in range(n_samples):
        if i < n_samples // 2:
            words = np.random.choice(spam_words, np.random.randint(5, 20))
            text = ' '.join(words) + ' ' + ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(10, 50)))
            labels.append('spam')
        else:
            words = np.random.choice(ham_words, np.random.randint(8, 25))
            text = ' '.join(words) + ' ' + ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(20, 100)))
            labels.append('ham')
        texts.append(text)
    
    df = pd.DataFrame({'text': texts, 'label': labels})

print(f"\nDataset Info:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")

df.to_csv('email_spam_dataset.csv', index=False)
print(f"\nDataset saved as 'email_spam_dataset.csv'")
files.download('email_spam_dataset.csv')

plt.figure(figsize=(8, 6))
df['label'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Class Distribution')
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

print("Preprocessing text data...")
df['cleaned_text'] = df['text'].apply(clean_text)
print("Preprocessing completed!")

print("\nSample of cleaned text:")
print(df[['text', 'cleaned_text']].head())

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['cleaned_text'])
y = df['label'].map({'ham': 0, 'spam': 1})

print(f"\nFeature matrix shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42, probability=True)
}

results = {}
print("\n" + "="*60)
print("MODEL TRAINING AND EVALUATION")
print("="*60)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred
    }
    
    print(f"{name} Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

cv_scores = {}
print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS")
print("="*60)

for name, model in models.items():
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores[name] = cv_score
    print(f"\n{name} - 5-Fold CV:")
    print(f"  Mean Accuracy: {cv_score.mean():.4f} (+/- {cv_score.std()*2:.4f})")
    print(f"  Individual Scores: {cv_score}")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, (name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    axes[idx].set_title(f'{name} - Confusion Matrix', fontsize=14)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
models_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in models_names]
precisions = [results[m]['precision'] for m in models_names]
recalls = [results[m]['recall'] for m in models_names]
f1_scores = [results[m]['f1'] for m in models_names]

x = np.arange(len(models_names))
width = 0.2

plt.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='skyblue')
plt.bar(x - 0.5*width, precisions, width, label='Precision', color='lightgreen')
plt.bar(x + 0.5*width, recalls, width, label='Recall', color='lightcoral')
plt.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='gold')

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance Comparison')
plt.xticks(x, models_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']
print("\n" + "="*60)
print(f"BEST MODEL: {best_model_name}")
print(f"F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
print("="*60)

test_emails = [
    "Congratulations! You've won a free iPhone! Click here to claim your prize now!",
    "Meeting tomorrow at 10 AM in conference room. Please bring the project reports.",
    "URGENT: Your account has been compromised. Verify your details immediately!",
    "Hi, can we reschedule our lunch meeting to Friday? Thanks!",
    "Get rich quick! Make $5000 per day working from home. Limited offer!",
]
print("\n" + "="*60)
print("TESTING WITH CUSTOM EMAILS")
print("="*60)

test_cleaned = [clean_text(email) for email in test_emails]
test_features = tfidf.transform(test_cleaned)
predictions = best_model.predict(test_features)
probabilities = best_model.predict_proba(test_features) if hasattr(best_model, 'predict_proba') else None

results_df = pd.DataFrame({
    'Email': test_emails,
    'Prediction': ['SPAM' if p == 1 else 'HAM' for p in predictions]
})

if probabilities is not None:
    results_df['Spam Probability'] = probabilities[:, 1]

print("\nTest Results:")
print(results_df.to_string(index=False))

for i, email in enumerate(test_emails):
    print(f"\nEmail {i+1}:")
    print(f"Text: {email[:100]}...")
    print(f"Prediction: {'SPAM' if predictions[i] == 1 else 'HAM'}")
    if probabilities is not None:
        print(f"Spam Probability: {probabilities[i][1]:.2%}")
    print("-" * 50)

feature_names = tfidf.get_feature_names_out()
if hasattr(best_model, 'coef_'):
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.coef_[0]
    })
    feature_importance = feature_importance.sort_values('importance', key=abs, ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    colors = ['red' if x < 0 else 'green' for x in feature_importance['importance'].head(10)]
    plt.barh(feature_importance['feature'].head(10), feature_importance['importance'].head(10), color=colors)
    plt.xlabel('Importance')
    plt.title(f'Top 20 Important Features - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Dataset Size: {len(df)} emails")
print(f"Features Extracted: {X.shape[1]}")
print(f"Best Model: {best_model_name}")
print(f"Overall Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"Overall F1-Score: {results[best_model_name]['f1']:.4f}")
print("\nModel Performance Summary:")
performance_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()]
})
print(performance_df.to_string(index=False))

print("\n" + "="*60)
print("SAVING THE BEST MODEL")
print("=",60)

import joblib
joblib.dump(best_model, 'best_spam_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
files.download('best_spam_model.pkl')
files.download('tfidf_vectorizer.pkl')
print("Best model and vectorizer saved and downloaded successfully!")

def predict_spam(email_text):
    cleaned = clean_text(email_text)
    features = tfidf.transform([cleaned])
    prediction = best_model.predict(features)[0]
    probability = best_model.predict_proba(features)[0][1] if hasattr(best_model, 'predict_proba') else None
    return "SPAM" if prediction == 1 else "HAM", probability

print("\n" + "="*60)
print("INTERACTIVE TESTING (Modify the email below)")
print("=",60)

test_single = "Your account has been locked. Click here to unlock immediately!"
result, prob = predict_spam(test_single)
print(f"\nTesting email: {test_single}")
print(f"Result: {result}")
if prob:
    print(f"Spam Confidence: {prob:.2%}")