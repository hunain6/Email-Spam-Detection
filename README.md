# Email Spam Detection

This project detects whether an email is spam or not spam (ham). It uses machine learning to classify emails.

## What this project does

- Takes email text as input
- Cleans and processes the text
- Uses multiple machine learning models to predict if email is spam
- Shows which model works best
- Saves the best model for future use

## Requirements

You need these Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- joblib

## Dataset

The code can work in two ways:

1. **Use your own dataset** - Upload a CSV file with two columns:
   - 'text' column containing email content
   - 'label' column containing 'spam' or 'ham'

2. **Create sample dataset** - If you don't upload a file, the code creates 10,000 sample emails (5000 spam, 5000 ham)

## How it works

### Step 1: Load or create data
- Reads your CSV file or creates sample data
- Shows class distribution (how many spam vs ham emails)
- Saves dataset as 'email_spam_dataset.csv'

### Step 2: Clean the text
- Converts all text to lowercase
- Removes numbers and special characters
- Removes common words (like 'the', 'and', 'is')
- Reduces words to their root form (like 'running' becomes 'run')

### Step 3: Convert text to numbers
- Uses TF-IDF to turn text into numbers that computers can understand
- Looks at single words and pairs of words
- Limits to 5000 most important features

### Step 4: Train models
Trains four different models:
- Naive Bayes
- Logistic Regression
- Random Forest
- SVM

### Step 5: Test and compare
- Splits data into 80% training and 20% testing
- Shows accuracy, precision, recall, and F1-score for each model
- Performs 5-fold cross-validation

### Step 6: Visualize results
Shows these graphs:
- Class distribution (spam vs ham count)
- Confusion matrices for each model
- Performance comparison chart
- ROC curves

### Step 7: Test with new emails
You can test with custom emails. Example emails in the code:
- "Congratulations! You've won a free iPhone..."
- "Meeting tomorrow at 10 AM..."

### Step 8: Save best model
- Saves the best performing model as 'best_spam_model.pkl'
- Saves the TF-IDF vectorizer as 'tfidf_vectorizer.pkl'
- Downloads both files to your computer

## How to use

### In Google Colab:
1. Copy the code into a new Colab notebook
2. Run all cells
3. If you have your own dataset, upload when prompted
4. Wait for training to complete
5. Check the results and graphs
6. Test with your own emails at the end

### To predict a single email after training:
```python
result, confidence = predict_spam("Your email text here")
print(f"This email is: {result}")
print(f"Confidence: {confidence:.2%}")
