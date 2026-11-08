
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


**Models Used**
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbor (KNN)
**Naive Bayes**
Support Vector Machine (SVM)


**## Model Comparison Results**

Logistic Regression
Accuracy: 0.8146
AUC: 0.9299
Precision: 0.7638
Recall: 0.9238
F1 Score: 0.8362
MCC: 0.6422

Decision Tree
Accuracy: 0.9854
AUC: 0.9857
Precision: 1.0000
Recall: 0.9714
F1 Score: 0.9855
MCC: 0.9712

KNN
Accuracy: 0.6976
AUC: 0.8335
Precision: 0.7129
Recall: 0.6857
F1 Score: 0.6990
MCC: 0.3956

Naive Bayes
Accuracy: 0.8293
AUC: 0.9043
Precision: 0.8062
Recall: 0.8762
F1 Score: 0.8402
MCC: 0.6602

SVM
Accuracy: 0.7171
AUC: 0.7477
Precision: 0.6983
Recall: 0.7905
F1 Score: 0.7411
MCC: 0.4360


**Model Observations**

**Logistic Regression**

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

