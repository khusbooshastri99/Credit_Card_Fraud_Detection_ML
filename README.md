# Credit_Card_Fraud_Detection

# What is credit card fraud.
  
  Credit card fraud is a wide-ranging term for theft and fraud committed using or involving a payment card, such as a credit card or debit card as a fraudulent source of funds in a transaction. It is important to detect such fraud via some novel methods.
  Here we will apply several Machine Learning algorithms to make predictions.

# Data Source: https://www.kaggle.com/mlg-ulb/creditcardfraud

It is a CSV file, containing 31 features and the last feature is used to classify the transaction whether it is a fraud or legit.

# Information about the data set
  
  1. The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 146 frauds out of 284,807 transactions. The dataset is highly unbalanced.
  2. It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
  3. The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.
  
# Flow of Project

We have done Exploratory Data Analysis on full data. And as the dataset is highly unbalanced so we are under sampling i.e. Building a sample dataset containing similar distribution of Normal Transactions and Fraudulent Transactions.
And then finally we have used Logistic Regression Technique to predict, to train, to test the data and to predict whether the transaction is Fraud or legit.

# How to Run the Project

In order to run the project download the data from the above mentioned source.

# Prerequisites

You just need to have the Google Collaboratory in your machine.

# Authors

Khushboo Shastri - Complete work

# Acknowledgement

Applied AI Course
