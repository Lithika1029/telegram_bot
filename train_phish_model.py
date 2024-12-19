import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the dataset
dataset_path = 'url_spam_classification.csv'  # Replace with your file path
df = pd.read_csv(dataset_path)

# Step 2: Preprocess the data
print("Initial dataset size:", df.shape)

# Normalize and clean 'is_spam' column
df['is_spam'] = df['is_spam'].astype(str).str.strip().str.lower()
df['is_spam'] = df['is_spam'].map({'true': 1, 'false': 0})  # Map 'true' to 1 and 'false' to 0

# Drop invalid rows where mapping failed (NaN values)
df = df.dropna(subset=['is_spam'])

# Confirm cleaning results
print("Cleaned dataset size:", df.shape)
print("Target value counts:\n", df['is_spam'].value_counts())

# Step 3: Split the data
X = df['url']
y = df['is_spam']

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Vectorize the URLs using TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))  # Trigrams to 5-grams
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train a Logistic Regression model
model = LogisticRegression(max_iter=200)  # Increase max_iter for convergence
model.fit(X_train_tfidf, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_tfidf)

# Print Accuracy and Classification Report
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save the model and vectorizer for deployment
joblib.dump(model, 'phishing_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("\nModel and vectorizer saved successfully!")
print("Final dataset size:", df.shape)
