
#Cell#1
########################################################################

# TO DO,
# get rid of max null columns,
# Will be using 
# Map categorical to dummy df / one hot encoder / labelencode,
# Map binary categorical data to 0 and 1,
# Mapping letter \"l\" and \"o\" to numbers 1 and 0,

# Deal with weird space character s53 (delete or encode)
# Correlation between two columns

# https://www.analyticsvidhya.com/blog/2015/11/8-ways-deal-continuous-variables-predictive-modeling/



#Declare all libraries here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statmodels.api as sm

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()




#Cell#2
#######################################################################

# Read in Training Data
df = pd.read_csv('train.csv')



#Cell#3
#######################################################################

#Checking Datatypes and Null Counts of original data
df.info()
df.isnull().sum()


#Cell#4
#######################################################################

# Viewing all the unique counts of each value in each column
column_names = df.columns

for column in column_names:
    print(df[column].value_counts())
    print("\n************\n")


#Cell#5
#######################################################################

# Identify and drop majority null columns, 
# As they have more than 70% missing values and this would have no significant effect on the target column

majority_null = ['s59','s57','s56','s55','s54']
df.drop(majority_null, axis = 1, inplace= True)



#Cell#6
#######################################################################

# Use Label encoding for binary categorical data 
# Using scikit.learn library (Only two options eg: M/F, Y/N)

# Creating instance of label encoder from scikit.learn
label_encoder = preprocessing.LabelEncoder()


# columns to encode
label_encode = ['gender','s11','s12','s53','s58']

for column in label_encode:
    df[column]= label_encoder.fit_transform(df[column])  
    df[column].unique()
    df[column].value_counts()




#Cell#7
#######################################################################

# Use one-hot encoding instead of label encoding for non-binary categories to avoid ranking / bias   
# using get_dummies from pandas library

# columns to encode
one_hot_encode = ['s16','s17','s18','s69','s70','s71']
ohe_df = pd.get_dummies(df[one_hot_encode])

# Drop column each of the original columns in the original database 
# as they are now encoded
df = df.drop(columns=one_hot_encode, axis = 1)

# Join the encoded df
df = df.join(ohe_df)



#Cell#8
#######################################################################

#Finding statistical corelation between the variables using multiple regression analysis
#Ordinary least square function

mod = sm.ols(formula='Label', data=df)