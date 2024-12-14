#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Simulated job dataset
data = {
    'job_description': [
        "Looking for Python developer with machine learning expertise",
        "Marketing manager needed with experience in digital advertising",
        "Nurse with experience in critical care and patient management",
        "Full-stack developer skilled in React, Node.js, and databases",
        "SEO specialist required for content optimization and strategy",
        "Data scientist proficient in Python, R, and statistical analysis"
    ],
    'category': ["IT", "Marketing", "Healthcare", "IT", "Marketing", "IT"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the dataset
print(df)


# In[3]:


# Define X (job descriptions) and y (categories)
X = df['job_description']
y = df['category']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_tfidf = vectorizer.fit_transform(X)

# Check dimensions of X_tfidf
print("TF-IDF Shape:", X_tfidf.shape)


# In[4]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Train Naive Bayes Classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Train SVM Classifier
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate the models
nb_predictions = nb_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)

print("Naive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions))

print("SVM Classification Report:")
print(classification_report(y_test, svm_predictions))


# In[5]:


# User profile input
user_profile = ["Python developer with experience in data science and machine learning"]

# Convert user profile and job descriptions into TF-IDF vectors
user_tfidf = vectorizer.transform(user_profile)
job_tfidf = vectorizer.transform(X)

# Calculate cosine similarity
cosine_sim = cosine_similarity(user_tfidf, job_tfidf)

# Rank jobs based on similarity scores
job_indices = np.argsort(cosine_sim[0])[::-1]  # Sort in descending order
print("Top 3 Job Recommendations:")

# Display top 3 job matches
for idx in job_indices[:3]:
    print(f"Job Description: {X.iloc[idx]} | Category: {y.iloc[idx]}")


# In[6]:


# Dummy relevance scores for evaluation (1: relevant, 0: not relevant)
# In a real-world setting, this will be user-labeled relevance
relevance_scores = [[1, 0, 1, 0, 0, 1]]  # Example relevance

def mean_average_precision(relevance_scores):
    average_precisions = []
    for scores in relevance_scores:
        num_relevant = 0
        avg_precision = 0
        for i, score in enumerate(scores):
            if score == 1:
                num_relevant += 1
                avg_precision += num_relevant / (i + 1)
        if num_relevant > 0:
            average_precisions.append(avg_precision / num_relevant)
    return np.mean(average_precisions)

# Calculate MAP
map_score = mean_average_precision(relevance_scores)
print("Mean Average Precision (MAP):", map_score)

