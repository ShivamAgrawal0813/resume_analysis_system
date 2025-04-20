import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import pickle

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Sample data for resume categorization
categories = ['HR', 'Technology', 'Marketing', 'Finance', 'Sales', 'Engineering', 'Healthcare', 'Design']
resumes = [
    "Human resources manager with experience in recruitment and employee relations",
    "Software engineer with Java and Python experience building web applications",
    "Marketing specialist with social media and SEO expertise",
    "Financial analyst experienced in budgeting and forecasting",
    "Sales representative with strong closing skills and customer relationships",
    "Mechanical engineer with 3D modeling and product design experience",
    "Registered nurse with patient care and clinical experience",
    "Graphic designer with Adobe Creative Suite skills and UI/UX experience"
]

# Create DataFrame
df_categorization = pd.DataFrame({
    'resume_text': resumes,
    'category': categories
})

# TF-IDF Vectorization and train categorization model
tfidf_vectorizer_categorization = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer_categorization.fit_transform(df_categorization['resume_text'])
y = df_categorization['category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train categorization model
rf_classifier_categorization = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_categorization.fit(X_train, y_train)

# Sample data for job recommendation
jobs = ['Data Scientist', 'Web Developer', 'Product Manager', 'Financial Analyst', 
        'UI Designer', 'Network Engineer', 'Marketing Specialist', 'HR Manager']
job_resumes = [
    "Data analysis with Python, statistics, machine learning, SQL experience",
    "Front-end development with HTML, CSS, JavaScript, React, and backend with Node.js",
    "Product management with agile methodologies, roadmap planning, and stakeholder management",
    "Financial modeling, budget analysis, forecasting, Excel and accounting expertise",
    "User interface design with Figma, Adobe XD, wireframing, prototyping, and user research",
    "Network configuration, security implementation, troubleshooting, Cisco certification",
    "Marketing campaigns, social media management, content creation, SEO expertise",
    "Recruitment, employee relations, performance management, HR policies implementation"
]

# Create DataFrame for job recommendation
df_job = pd.DataFrame({
    'resume_text': job_resumes,
    'job': jobs
})

# TF-IDF Vectorization and train job recommendation model
tfidf_vectorizer_job_recommendation = TfidfVectorizer(max_features=1000)
X_job = tfidf_vectorizer_job_recommendation.fit_transform(df_job['resume_text'])
y_job = df_job['job']

# Split data
X_job_train, X_job_test, y_job_train, y_job_test = train_test_split(X_job, y_job, test_size=0.2, random_state=42)

# Train job recommendation model
rf_classifier_job_recommendation = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_job_recommendation.fit(X_job_train, y_job_train)

# Save models to disk
print("Saving models to disk...")
pickle.dump(rf_classifier_categorization, open('models/rf_classifier_categorization.pkl', 'wb'))
pickle.dump(tfidf_vectorizer_categorization, open('models/tfidf_vectorizer_categorization.pkl', 'wb'))
pickle.dump(rf_classifier_job_recommendation, open('models/rf_classifier_job_recommendation.pkl', 'wb'))
pickle.dump(tfidf_vectorizer_job_recommendation, open('models/tfidf_vectorizer_job_recommendation.pkl', 'wb'))

print("Models generated successfully!") 