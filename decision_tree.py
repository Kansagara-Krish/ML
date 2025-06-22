import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import numpy as np

# Load the dataset
df = pd.read_csv(r"C:\Users\kansa\OneDrive\Desktop\ML\Projects\car_data.csv")

print(df.isnull().sum())

print(df.head())
gender={'Male':0,'Female':1}

df['Gender']=df['Gender'].map(gender)

x_columns=['Gender','Age','AnnualSalary']
y=df['Purchased']
X=df[x_columns]

print(X.head())
print(y.head())
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the model

dtree=DecisionTreeClassifier()
dtree=dtree.fit(X_train,y_train)

input_data=np.array([1,50,109500]).reshape(1,-1)
print("[1] means Purchased [0] means not purchased")
print(dtree.predict(input_data))
#Make predictions



# Scale the features
