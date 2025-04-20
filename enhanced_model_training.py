"""
Enhanced model training script using the actual datasets from archive(1)
This script trains resume categorization and job recommendation models using real data.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import os
import pickle
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

print("Starting enhanced model training...")
start_time = time.time()

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+\s', ' ', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Load and prepare the resume categorization dataset
print("Loading resume categorization dataset...")
resume_data_path = 'archive (1)/clean_resume_data.csv'
resume_df = pd.read_csv(resume_data_path)

print(f"Dataset shape: {resume_df.shape}")
print(f"Categories: {resume_df['Category'].unique()}")

# Clean the Feature column
print("Cleaning resume text...")
resume_df['CleanedFeature'] = resume_df['Feature'].apply(clean_text)

# Split data for resume categorization
X = resume_df['CleanedFeature']
y = resume_df['Category']

# Split into train and test sets
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization for resume categorization
print("Performing TF-IDF vectorization for resume categorization...")
tfidf_vectorizer_categorization = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer_categorization.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer_categorization.transform(X_test)

# Train RandomForest classifier for resume categorization
print("Training RandomForest classifier for resume categorization...")
rf_classifier_categorization = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_classifier_categorization.fit(X_train_tfidf, y_train)

# Evaluate categorization model
print("Evaluating resume categorization model...")
y_pred = rf_classifier_categorization.predict(X_test_tfidf)
categorization_accuracy = accuracy_score(y_test, y_pred)
print(f"Resume Categorization Accuracy: {categorization_accuracy:.4f}")

# Sample job descriptions for job recommendation
# We'll create a diverse set of job descriptions for various roles
print("Creating job recommendation model...")

# Since we couldn't access the jobs dataset, we'll use our predefined jobs
from job_descriptions import JOB_DESCRIPTIONS

# Create a DataFrame from our job descriptions
job_titles = []
job_descriptions_text = []
job_skills = []

for job in JOB_DESCRIPTIONS:
    job_titles.append(job['title'])
    # Combine description and skills into a single text
    full_description = job['description'] + " " + " ".join(job['skills_required'])
    job_descriptions_text.append(full_description)
    job_skills.append(", ".join(job['skills_required']))

jobs_df = pd.DataFrame({
    'JobTitle': job_titles,
    'JobDescription': job_descriptions_text,
    'RequiredSkills': job_skills
})

# Clean job descriptions
jobs_df['CleanedJobDescription'] = jobs_df['JobDescription'].apply(clean_text)

# Split data for job recommendation
X_job = jobs_df['CleanedJobDescription']
y_job = jobs_df['JobTitle']

# Split into train and test sets
X_job_train, X_job_test, y_job_train, y_job_test = train_test_split(
    X_job, y_job, test_size=0.2, random_state=42
)

# TF-IDF vectorization for job recommendation
print("Performing TF-IDF vectorization for job recommendation...")
tfidf_vectorizer_job_recommendation = TfidfVectorizer(max_features=5000, stop_words='english')
X_job_train_tfidf = tfidf_vectorizer_job_recommendation.fit_transform(X_job_train)
X_job_test_tfidf = tfidf_vectorizer_job_recommendation.transform(X_job_test)

# Train RandomForest classifier for job recommendation
print("Training RandomForest classifier for job recommendation...")
rf_classifier_job_recommendation = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_classifier_job_recommendation.fit(X_job_train_tfidf, y_job_train)

# Evaluate job recommendation model
print("Evaluating job recommendation model...")
y_job_pred = rf_classifier_job_recommendation.predict(X_job_test_tfidf)
job_recommendation_accuracy = accuracy_score(y_job_test, y_job_pred)
print(f"Job Recommendation Accuracy: {job_recommendation_accuracy:.4f}")

# Save models to disk
print("Saving models to disk...")
pickle.dump(rf_classifier_categorization, open('models/rf_classifier_categorization.pkl', 'wb'))
pickle.dump(tfidf_vectorizer_categorization, open('models/tfidf_vectorizer_categorization.pkl', 'wb'))
pickle.dump(rf_classifier_job_recommendation, open('models/rf_classifier_job_recommendation.pkl', 'wb'))
pickle.dump(tfidf_vectorizer_job_recommendation, open('models/tfidf_vectorizer_job_recommendation.pkl', 'wb'))

end_time = time.time()
print(f"Model training and saving completed in {(end_time - start_time):.2f} seconds!")
print("Models are now ready to use in the Resume Analysis System.") 