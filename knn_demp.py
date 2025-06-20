from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

df=pd.read_csv(r"C:\Users\kansa\OneDrive\Desktop\ML\KNN_demo.csv")

print(df.shape)
print(df.isnull().sum())

df.drop("Unnamed: 32",axis=1,inplace=True)
df.drop('id',axis=1,inplace=True)
print(df.dtypes)

y=df["diagnosis"]
x=df.drop("diagnosis",axis=1)

#Imporve result use scaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)

#Split the dataset
#Train the dataset

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Train the model

K=[]
testing=[]
best_score=0
best_k=1
k_score=[]
training=[]
score={}

for k in range(2,21):
    clf=KNeighborsClassifier(n_neighbors=k)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    acc=accuracy_score(y_test,y_pred)

    if acc>best_score:
        best_score=acc
        best_k=k
        k_score.append(acc)

    train=clf.score(x_train,y_train)
    test=clf.score(x_test,y_test)

    K.append(k)
    training.append(train)
    testing.append(test)
    print(train,":",test)

print(f"Best score is {best_score:0.4f}")
print("Value of k is",best_k)
#Draw stripplot for taining and tetsing 


ax=sb.stripplot(x=K,y=training)
ax.set(xlabel="Value of k",ylabel="training")
plt.show()

ax=sb.stripplot(x=K,y=testing)
ax.set(xlabel="Value of k",ylabel="testing")
plt.show()

plt.scatter(K,training,color='k')
plt.scatter(K,testing,color='r')
plt.show()

#checking of new data input and predict

new_input = [[17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 
              1.095, 0.9053, 8.589, 153.4, 0.0064, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
              25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]

print(len(new_input[0]))

#Scaler the data so that it can give more accurate result
new_input-scaler.fit_transform(new_input)

clf=KNeighborsClassifier(n_neighbors=best_k)
clf.fit(x_train,y_train)

new_pred=clf.predict(new_input)
print(new_pred)
