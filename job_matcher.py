"""
Job matching module for calculating match scores between resumes and job descriptions.
Uses TF-IDF vectorization and cosine similarity for content-based matching,
along with specific skill and education matching.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from job_descriptions import JOB_DESCRIPTIONS

def calculate_match_score(resume_text, extracted_skills, education, job):
    """
    Calculate a match score between a resume and a job description.
    Returns a score between 0 and 100.
    
    Parameters:
    - resume_text: The cleaned resume text
    - extracted_skills: List of skills extracted from the resume
    - education: List of education fields extracted from the resume
    - job: Job description dictionary
    
    Returns:
    - match_score: Integer between 0 and 100
    - match_details: Dictionary with details about the match
    """
    # Initialize scores for different components
    skill_score = 0
    education_score = 0
    description_score = 0
    
    # Calculate skill match score (50% of total)
    if extracted_skills and job["skills_required"]:
        matching_skills = [skill for skill in extracted_skills if skill in job["skills_required"]]
        skill_score = len(matching_skills) / len(job["skills_required"]) * 50
        matching_skill_percentage = len(matching_skills) / len(job["skills_required"]) * 100
    else:
        matching_skills = []
        matching_skill_percentage = 0
    
    # Calculate education match score (20% of total)
    if education and job["education"]:
        matching_education = [edu for edu in education if any(job_edu.lower() in edu.lower() for job_edu in job["education"])]
        education_score = len(matching_education) / len(job["education"]) * 20
    
    # Calculate description similarity score (30% of total)
    # Combine resume text and job description
    texts = [resume_text, job["description"]]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        # Transform texts to TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Scale similarity to a 0-30 score
        description_score = similarity * 30
    except:
        # Fallback if vectorization fails
        description_score = 15  # Default middle score
    
    # Calculate total match score
    total_score = skill_score + education_score + description_score
    
    # Round to nearest integer
    match_score = round(total_score)
    
    # Cap at 100
    match_score = min(match_score, 100)
    
    # Prepare match details
    match_details = {
        "matching_skills": matching_skills,
        "matching_skill_percentage": round(matching_skill_percentage),
        "skill_score": round(skill_score),
        "education_score": round(education_score),
        "description_score": round(description_score),
        "total_score": match_score
    }
    
    return match_score, match_details

def get_job_matches(resume_text, extracted_skills, education, top_n=3):
    """
    Get the top N job matches for a resume.
    
    Parameters:
    - resume_text: The cleaned resume text
    - extracted_skills: List of skills extracted from the resume
    - education: List of education fields extracted from the resume
    - top_n: Number of top matches to return
    
    Returns:
    - matches: List of dictionaries with job matches and scores
    """
    matches = []
    
    for job in JOB_DESCRIPTIONS:
        score, details = calculate_match_score(resume_text, extracted_skills, education, job)
        
        match = {
            "job_id": job["id"],
            "job_title": job["title"],
            "company": job["company"],
            "match_score": score,
            "match_details": details,
            "location": job["location"]
        }
        
        matches.append(match)
    
    # Sort matches by score in descending order
    matches.sort(key=lambda x: x["match_score"], reverse=True)
    
    # Return top N matches
    return matches[:top_n] 