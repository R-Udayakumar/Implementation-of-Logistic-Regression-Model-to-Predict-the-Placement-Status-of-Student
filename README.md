# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the python library pandas
2. Read the dataset of Placement_Data
3. Copy the dataset in data1
4. Remove the columns which have null values using drop()
5. Import the LabelEncoder for preprocessing of the dataset
6. Assign x and y as status column values
7. From sklearn library select the model to perform Logistic Regression
8. Print the accuracy, confusion matrix and classification report of the dataset

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: UDAYAKUMAR R
RegisterNumber:  212222230163
*/
```
```python
import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1) # Removes the specified row or column
data1.head()

data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x
y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression (solver ='liblinear') # A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred) # Accuracy Score = (TP+TN)/ (TP+FN+TN+FP) ,True +ve/
#accuracy_score (y_true,y_pred, normalize = false)
# Normalize : It contains the boolean value (True/False). If False, return the number of correct
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
## PLACEMENT DATA 
![Screenshot from 2023-09-14 08-57-59](https://github.com/R-Udayakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708024/4ab1b9df-5c42-4d39-a173-3b0585d5b6c4)
## SALARY DATA
![image](https://github.com/R-Udayakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708024/7ec04afd-685c-48ac-bf1e-47b6139febd2)
## CHECKING THE NULL() FUNCTION
![image](https://github.com/R-Udayakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708024/cc2090e7-b25c-403c-8809-4b0f0722d5be)
## DATA DUPLICATE
![image](https://github.com/R-Udayakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708024/587731a4-0a14-4630-be95-b1078b2fdd5d)
## PRINT DATA
![image](https://github.com/R-Udayakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708024/91cc2a17-0588-4df6-a220-2cd351959b2c)
## DATA-STATUS
![image](https://github.com/R-Udayakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708024/4fd17606-20e3-423d-ab18-235e15ba72c1)
## Y_PREDICTION ARRAY
![image](https://github.com/R-Udayakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708024/16eae5b4-dd79-4234-9fd1-811916200754)
## ACCURACY VALUE
![image](https://github.com/R-Udayakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708024/f0e21144-404c-46b2-a483-26cb4b051360)
## CONFUSION ARRAY
![image](https://github.com/R-Udayakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708024/86577f25-0adb-4b7c-9774-7156ce23601c)
## CLASSIFICATION REPORT
![image](https://github.com/R-Udayakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708024/7cc3e320-00e1-49d6-b498-53f8f7b933b1)
## PREDICTION OF LR
![image](https://github.com/R-Udayakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708024/afe7f64a-d94e-473c-a59a-6187adcdcb64)
![image](https://github.com/R-Udayakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708024/9ac8f2ce-22df-46d4-bb24-36f50cfac3dd)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
