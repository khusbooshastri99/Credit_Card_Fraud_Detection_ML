#Importing the Dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Importing the Dataset

credit_card_data = pd.read_csv("creditcard.csv")

# credit card information
credit_card_data.info()

# Checking the number of missing values in each column
credit_card_data.isnull().sum()

#distribution of legit transactions and fraudulent transactions
credit_card_data['Class'].value_counts()

#0.0    47481
#1.0      146
#Name: Class, dtype: int64

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

#	      Time	        V1	     V2	      V3	       V4	        V5	     V6	       V7	       V8	      V9	        V10	     V11	      V12	     V13	   V14	      V15	      V16	      V17	       V18	    V19	      V20	      V21	      V22	       V23	    V24	      V25	      V26	      V27	      V28	      Amount
#Class																														
#0.0	 28187.518523	-0.217647	0.004622	0.729134	0.173144	-0.232960	0.109679	-0.095198	0.041927	0.152781	-0.040902	0.366836	-0.338466	0.172874	0.207887	0.115604	0.003736	0.150373	-0.085585	-0.031096	0.046262	-0.029852	-0.106718	-0.038640	0.008178	0.135952	0.022116	0.003026	0.003920	91.749512
#1.0	 26650.410959	-7.748350	5.502036	-10.507153	6.017631	-5.734345	-2.290523	-8.141188	3.800624	-3.626437	-7.597215	5.466621	-8.740580	0.377715	-8.954239	0.123910	-5.926464	-9.975972	-3.691881	0.912631	0.471417	0.881448	-0.228124	-0.286066	-0.085893	0.253943	0.167237	0.616539	0.037065	100.769589

#Under Sampling

#Build a sample dataset containing similar distribution of Normal Transactions and Fraudulent Transactions

#Number of Fraudulent Transactions --> 146

legit_sample = legit.sample(n=146)

#Concatenating Two DataFrames

new_dataset = pd.concat([legit_sample, fraud],axis=0)
new_dataset.head()
new_dataset.tail()

new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()

#	    Time	        V1	     V2	       V3	       V4	         V5	      V6	      V7	     V8	       V9	        V10	      V11	      V12	    V13	     V14	      V15	       V16	    V17	       V18	     V19	    V20	     V21	      V22	        V23	     V24	     V25	      V26	      V27	     V28	     Amount
#Class																														
#0.0	27053.356164	-0.174046	-0.279267	0.711947	0.354025	-0.320821	0.038357	0.129092	0.058761	0.158889	-0.121767	0.489076	-0.332138	0.103310	0.151404	0.312262	0.083835	0.191963	-0.058800	-0.002191	0.246812	0.034861	-0.064385	-0.084009	0.071056	0.130332	-0.067058	-0.015625	0.028539	173.220411
#1.0	26650.410959	-7.748350	5.502036	-10.507153	6.017631	-5.734345	-2.290523	-8.141188	3.800624	-3.626437	-7.597215	5.466621	-8.740580	0.377715	-8.954239	0.123910	-5.926464	-9.975972	-3.691881	0.912631	0.471417	0.881448	-0.228124	-0.286066	-0.085893	0.253943	0.167237	0.616539	0.037065	100.769589

#Splitting the data into Features & Targets

X = new_dataset.drop(columns = 'Class', axis=1)
Y = new_dataset['Class']

#Split the data into Training data & Testing data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#(292, 30) (233, 30) (59, 30)


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

#Accuracy on training data: 1.0

# Accuracy on test data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data:', testing_data_accuracy)

#Accuracy on test data: 0.9666666666666667
