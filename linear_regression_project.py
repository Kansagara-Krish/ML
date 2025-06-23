# This is a linear_regression_project python file created by your personal assistant

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
# Load the dataset
try:
    df =pd.read_csv(r"C:\Users\kansa\OneDrive\Desktop\ML\Projects\Student_Performance.csv")

    # Display the first few rows of the dataset
    print("First few row of dataset:")
    print(df.head())

    print("Information abou the data set")
    print(df.info())

    print("Descriptive statistics of the dataset:")
    print(df.describe())

    #check for missing values
    print("Checking for missing values:")
    print(df.isnull().sum())

    # Split the dataset into features and target variable
    X = df.drop('Performance Index', axis=1)
    y=df['Performance Index']

    print(X.head(),y.head())

    print(X.shape, y.shape)

    #visiualize the data
    
    df["Extracurricular Activities"].value_counts().plot.pie(autopct='%1.1f%%')
    plt.show()

    #Split the data in categorical and numerical data
    category=[i for i in X.columns if df[i].dtypes=="object"]
    numerical=[i for i in X.columns if df[i].dtypes!="object"]

    X=pd.get_dummies(X, columns=category, drop_first=True)
    #Train the data

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #Scale the data

    x_train= StandardScaler().fit_transform(x_train)
    x_test= StandardScaler().transform(x_test)
    #Train the model

    reg=LinearRegression()
    reg.fit(x_train, y_train)

    # Make predictions

    pred=reg.predict(x_test)
    print("Predictions:")
    print(pred)

    print("Score od train data",reg.score(x_train, y_train))

    print("Score of test data",reg.score(x_test, y_test))

    print("Coefficients of the model:",reg.coef_)

    print("Intercept of the model:",reg.intercept_)

    #predict new output of a data

 
    print("Prediction for new data:", new_prediction)


except Exception as e:
    print(e)
