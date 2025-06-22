import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv(r"C:\Users\kansa\OneDrive\Desktop\ML\Projects\adult.csv")

print(df.info())

print(df.isnull().sum())
try:
    y=df["income"]
    x=df.drop('income',axis=1) 
    categoty=[i for i in x.columns if x[i].dtype=='O']
    print(x.isnull().sum())

#encode the x data into numerical data
    x=pd.get_dummies(x,columns=categoty,drop_first=False)

    print(x.head())

    print(x.describe())

#Train the data
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Scaler for better result

    scaler=StandardScaler()

    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

#Train the model

    K=[]
    training=[]
    resting=[]
    best_score=0
    best_k=1
    n=0
    for k in range(2,21):
        clf=KNeighborsClassifier(n_neighbors=k)
        clf.fit(x_train,y_train)

        train=clf.score(x_train,y_train)
        test=clf.score(x_test,y_test)
        K.append(k)
        y_pred=clf.predict(x_test)
        acc=accuracy_score(y_test,y_pred)
        n=n+1
        print(n,".",train,":",test)

        if acc>best_score:
            best_score=acc
            best_k=k

    print(y,x)

    print("Best K:",best_k,"Best Score:",best_score)

    new_sample = pd.DataFrame([{
    'age': 48,
    'workclass': 'Private',
    'fnlwgt': 279724,
    'education': 'HS-grad',
    'education-num': 9,
    'marital-status': 'Married-civ-spouse',
    'occupation': 'Machine-op-inspct',
    'relationship': 'Husband',
    'race': 'White',
    'sex': 'Male',
    'capital-gain': 3103,
    'capital-loss': 0,
    'hours-per-week': 48,
    'native-country': 'United-States'
    }])

    new_data=new_sample.reindex(columns=x.columns,fill_value=0)

    new_data=scaler.fit_transform(new_data)

    acc=clf.predict(new_data)

    print(acc)

    #seaborn graph

    ax=sb.stripplot(x=K,y=x_train)
    ax.set(xlabel="Value of k",ylabel="x_train")
    plt.show()

    ax=sb.stripplot(x=K,y=y_train)
    ax.set(xlabel="Value of k",ylabel="y_train")
    plt.show()

    ax=sb.stripplot(x=K,y=x_test)
    ax.set(xlabel="Value of k",ylabel="x_test")
    plt.show()

    ax=sb.stripplot(x=K,y=y_test)
    ax.set(xlabel="Value of k",ylabel="y_test")
    plt.show()

    #Prediction

    plt.scatter(k,y_test)
    plt.scatter(k,x_test)
    plt.show()

except Exception as e:
    print(e)