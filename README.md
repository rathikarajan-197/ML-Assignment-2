
**Problem Statement**

The objective of this project is to predict whether a patient has heart disease using machine learning classification algorithms and compare their performance using various evaluation metrics.

**Dataset Description**

Dataset Name: Heart Disease Dataset

Source: Kaggle

Number of Records: 1025

Number of Features: 13

**Target Variable:**

0 = No Heart Disease
1 = Heart Disease

**Features Used:**

age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal

**GitHub Repository Link**
https://github.com/rathikarajan-197/ML-Assignment-2

**Streamlit Application Link**
https://ml-assignment-2-hoqzrthexnkwmi4g9yvzkp.streamlit.app/
Models Used
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbor (KNN)
**Naive Bayes**
Support Vector Machine (SVM)

**Model Comparison Table**
ML Model Name	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.8146	0.9299	0.7638	0.9238	0.8362	0.6422
Decision Tree	0.9854	0.9857	1.0000	0.9714	0.9855	0.9712
KNN	0.6976	0.8335	0.7129	0.6857	0.6990	0.3956
Naive Bayes	0.8293	0.9043	0.8062	0.8762	0.8402	0.6602
SVM	0.7171	0.7477	0.6983	0.7905	0.7411	0.4360


**Model Observations**

ML Model Name	Observation
Logistic Regression	Good baseline performance with high recall.
Decision Tree	Highest accuracy and best overall performance.
KNN	Lowest performance among all models.
Naive Bayes	Good balance between speed and accuracy.
SVM	Moderate performance.
Overall Winner	Decision Tree Classifier

Logistic Regression

Good baseline performance with high recall.

**Decision Tree**

Achieved the highest accuracy and best overall performance.

**KNN**

Lowest performance among all evaluated models.

**Naive Bayes**

Provided a good balance between speed and accuracy.

**SVM**

Moderate performance on this dataset.

**Overall Winner**

Decision Tree Classifier

**Conclusion**

Five machine learning classification models were implemented and evaluated using the Heart Disease Dataset. The performance of each model was measured using Accuracy, AUC, Precision, Recall, F1 Score, and MCC. Among all models, the Decision Tree Classifier achieved the highest accuracy of 98.54% and was identified as the best-performing model. The project was successfully implemented, tested, and deployed using Streamlit Community Cloud.

