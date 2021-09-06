#Importing the Dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Importing the Dataset

credit_card_data = pd.read_csv("creditcard.csv")

# credit card information
credit_card_data.info()

# Checking the number of missing values in each column
credit_card_data.isnull().sum()

#distribution of legit transactions and fraudulent transactions
credit_card_data['Class'].value_counts()

#This Dataset is highly unbalanced

#0--> Normal transaction

#1--> fraudulent

# Seperating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)

# Statical measures of the data
legit.Amount.describe()
fraud.Amount.describe()

## Compairing the values for both transactions
credit_card_data.groupby('Class').mean()

#Under Sampling

#Build a smple dataset containing similar distribution of Normal Transactions and Fraudulent Transactions

#Number of Fraudulent Transactions --> 146

legit_sample = legit.sample(n=146)

#Concatenating Two DataFrames

new_dataset = pd.concat([legit_sample, fraud],axis=0)
new_dataset.head()
new_dataset.tail()

new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()

#Splitting the data into Features & Targets

X = new_dataset.drop(columns = 'Class', axis=1)
Y = new_dataset['Class']

#Split the data into Training data & Testing data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#Model Training
#Logistic Regression
model = LogisticRegression()

# training the Logistic Regression Model with training data
model.fit(X_train, Y_train)

#Model Evalutation

#Accuracy Score
# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data:', training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data:', testing_data_accuracy)
