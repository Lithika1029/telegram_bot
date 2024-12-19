import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset (replace 'dataset.csv' with your actual file path)
data = pd.read_csv('new_data_urls.csv')

# Use a subset of the data for testing (adjust n as needed)
data = data.sample(n=10000, random_state=42)

# Preprocess the data
# Assuming the data has columns 'url' and 'status'
urls = data['url'].astype(str)  # Ensure URLs are strings
labels = data['status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(urls, labels, test_size=0.2, random_state=42)

# Feature extraction using TfidfVectorizer with optimized settings
vectorizer = TfidfVectorizer(
    analyzer='char_wb', 
    ngram_range=(3, 4),  # Reduced n-gram size for faster processing
    max_features=50000   # Limit the number of features
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # Use multiple cores
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer for later use
joblib.dump(model, 'phishing_detection_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
