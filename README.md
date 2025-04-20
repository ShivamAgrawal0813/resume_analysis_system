# Resume Analysis System

A machine learning-based system that analyzes resumes, categorizes them, recommends suitable jobs, and extracts key information.

## Features

- **Resume Categorization**: Classifies resumes into different professional categories
- **Job Recommendation**: Suggests suitable job roles based on resume content
- **Information Extraction**: Extracts name, email, phone, skills, and education from resumes
- **User-Friendly Interface**: Clean and intuitive web interface for easy interaction

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS
- **Machine Learning**: scikit-learn, TF-IDF, RandomForest
- **Data Processing**: pandas, numpy, regex
- **PDF Processing**: PyPDF2

## Project Structure

```
resume_analysis_system/
├── app.py                           # Main Flask application
├── generate_models.py               # Script to generate ML models
├── models/                          # Directory containing ML models
│   ├── rf_classifier_categorization.pkl
│   ├── tfidf_vectorizer_categorization.pkl
│   ├── rf_classifier_job_recommendation.pkl
│   └── tfidf_vectorizer_job_recommendation.pkl
├── templates/                       # HTML templates
│   └── resume.html
├── docs/                            # Documentation
│   ├── Resume_Analysis_System.pptx  # Presentation
│   └── Project_Report.docx          # Detailed report
└── sample_resumes/                  # Sample resume files for testing
```

## Installation and Setup

1. Clone this repository:
```
git clone https://github.com/yourusername/resume_analysis_system.git
cd resume_analysis_system
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Generate the machine learning models:
```
python generate_models.py
```

4. Run the Flask application:
```
python app.py
```

5. Access the web application:
```
http://127.0.0.1:5000
```

## Usage

1. Upload a resume in PDF or TXT format
2. The system will analyze the resume and display:
   - Resume category
   - Recommended job
   - Extracted information (name, contact, skills, education)

## Future Improvements

- Advanced NLP models (BERT, transformers)
- Skill gap analysis
- Resume scoring and enhancement suggestions
- Support for more file formats
- Integration with job boards

## Contributors

- [Your Name](https://github.com/yourusername)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 