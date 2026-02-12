# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, drop unnecessary columns, and encode categorical variables. 

2.Define the features (X) and target variable (y). 

3.Split the data into training and testing sets. 

4.Train the logistic regression model, make predictions, and evaluate using accuracy and other
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mohamed Ukkas R
RegisterNumber:  25007472
*/

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
```

## Output:
HEAD

<img width="1279" height="330" alt="384694261-6d3166e2-7fd9-4582-8729-4ba68a64036b" src="https://github.com/user-attachments/assets/b8702d74-3ede-4837-be36-792921eb9243" />

COPY

<img width="1141" height="345" alt="384694385-3e4ee2fa-f457-4c2c-a13a-d4791707c7d7" src="https://github.com/user-attachments/assets/74dddc49-b1ab-42ae-8584-cd2d73bb268e" />

FIT TRANSFORM

<img width="1122" height="707" alt="384694436-b0e4a287-35b1-4c2d-b386-d20f8f62b772" src="https://github.com/user-attachments/assets/52e5f2f5-4c6c-4e0b-a58e-2cfd3b408c1d" />

LOGISTIC REGRESSION

<img width="1231" height="309" alt="384694500-49664d5e-2913-456b-95fa-dd7ae5a15637" src="https://github.com/user-attachments/assets/42b529cc-e624-4717-98ce-9d476e032679" />


ACCURACY SCORE

<img width="1225" height="169" alt="384694533-0e91c61e-abcb-400f-a04f-c36ec2885d1c" src="https://github.com/user-attachments/assets/e672085f-b0bc-4efa-991d-2118773a3dde" />


CONFUSION MATRIX

<img width="1229" height="203" alt="388782587-58ce1bb2-1678-4407-9b13-cf33fabb3cc9" src="https://github.com/user-attachments/assets/1c87e61f-dff9-4ea2-b55b-8a248b1bca51" />


CLASSIFICATION REPORT & PREDICTION

<img width="1217" height="524" alt="384694700-f1591a15-9289-4a4e-a284-20d5b58298b3" src="https://github.com/user-attachments/assets/0ea789cf-5047-4f80-b7a4-68c6483061da" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
