'''What's the real problem here?
The company wants to avoid unfair or biased decisions when deciding salaries for new hires. Right now, humans might rely on gut feeling, which can lead to inconsistencies — even if two people have very similar skills and experience, they might get different salary offers. That’s the issue they want to fix.

What are they trying to achieve?
They want to build a system — probably using machine learning — that can look at data from previous hires (like experience, education, skills, etc.) and predict a fair salary for a new candidate. This model would:

Treat people with similar profiles equally

Reduce guesswork or bias in salary decisions

Standardize how salaries are offered during hiring

Why use data for this?
They likely have a lot of past data about job applicants, their qualifications, and the salaries they were offered. By analyzing this, the model can learn patterns — like how much salary was given to someone with 5 years of experience and a certain skill set — and apply that logic consistently.'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    # Load dataset
    df = pd.read_csv(r"C:\Users\kansa\OneDrive\Desktop\ML\Projects\expected_ctc.csv")
    
    print(df.head(10))
    
    print(df.info())
    
    print(df.describe())
    
    print(df.shape)
    
    print(df.isnull().sum())
    
    df.dropna(inplace=True)
    
    #drop IDX  and Applicant_ID column    
    df.drop(['IDX', 'Applicant_ID'], axis=1, inplace=True)

    # Target and features
    
    exp=np.array(df["Total_Experience"]).reshape(-1,1)
    c_ctc = np.array(df["Current_CTC"]).reshape(-1,1)
    e_ctc = np.array(df["Expected_CTC"])
    
    plt.subplot(2,2,1)
    plt.plot(exp,c_ctc,'*')
    plt.xlabel("Total Experience")
    plt.ylabel("Current CTC")
    plt.title("Total Experience  VS Current CTC")
    plt.grid()
    
    plt.subplot(2,2,2)
    plt.plot(exp,e_ctc,'*')
    plt.xlabel("Total Experience")
    plt.ylabel("Expected CTC")
    plt.title("Total Experience  VS Expected CTC")
    plt.grid()
    plt.show()
    
    df['Inhand_Offer'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6, 6))
    plt.title("Inhand Offer")
    plt.show()

    y = df['Expected_CTC']
    X = df.drop(['Expected_CTC'], axis=1)
    
    print(X.head())
    
    print(y.head())
    
    print(X.shape,y.shape)
    
    print(X.dtypes)
    
    print("-"*20)
    
    print(y.dtypes)

    print(X.value_counts())
    
    print(y.value_counts())   
    
    # Detect types
    categorical = [col for col in X.columns if X[col].dtype == 'object']
    numerical = [col for col in X.columns if X[col].dtype != 'object']
    
    # Outlier removal
    Q1 = X[numerical].quantile(0.25)
    Q3 = X[numerical].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X[numerical] < (Q1 - 1.5 * IQR)) | (X[numerical] > (Q3 + 1.5 * IQR))).any(axis=1)
    X, y = X[mask], y[mask]

    # Encode categorical
    X_encoded = pd.get_dummies(X, columns=categorical, drop_first=True)

    corr = df[numerical + ['Expected_CTC']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    
    df_melted = df[numerical].melt(var_name='Feature', value_name='Value')

    plt.figure(figsize=(14, 6))
    sns.boxplot(x='Feature', y='Value', data=df_melted, palette='Set2')
    plt.title('Boxplots of All Numerical Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    # Train-test split
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train[numerical] = scaler.fit_transform(X_train[numerical])
    X_test[numerical] = scaler.transform(X_test[numerical])
    '''
    # Model training
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Evaluation
    print("\nTrain R² Score:", reg.score(X_train, y_train))
    print("Test R² Score:", reg.score(X_test, y_test))

    coeff_df = pd.DataFrame({'Feature': X_encoded.columns, 'Coefficient': reg.coef_})
    print("\nTop 10 Influential Features:")
    print(coeff_df.reindex(coeff_df['Coefficient'].abs().sort_values(ascending=False).index).head(10))

    # New candidate input
    new_input = pd.DataFrame([
        [1, 1, 'NA', 'NA', 'NA', 'NA', 'NA', 'PG', 'Arts', 'Lucknow', 2020, None, None, None, None, None,
         'Guwahati', 'Pune', 1, 'N', 'NA', 1, 1, 1, 1],
        [23, 14, 'HR', 'Consultant', 'Analytics', 'H', 'HR', 'Doctorate', 'Chemistry', 'Surat', 1988, 'Others',
         'Surat', 1990, 'Chemistry', 'Mangalore', 1997, 'Bangalore', 'Nagpur', 2702664, 'Y',
         'Key_Performer', 2, 4, 1, 1]
    ], columns=X.columns)

    # Fill missing values without chained assignment
    for col in new_input.columns:
        if new_input[col].dtype == 'object':
            new_input[col] = new_input[col].fillna('None')
        else:
            new_input[col] = new_input[col].fillna(0)

    # Encoding and scaling for new input
    new_encoded = pd.get_dummies(new_input)
    new_encoded = new_encoded.reindex(columns=X_encoded.columns, fill_value=0)
    new_encoded[numerical] = scaler.transform(new_encoded[numerical])

    # Prediction
    new_predictions = reg.predict(new_encoded)
    print("\nPredicted Expected CTCs for new candidates:")
    for i, val in enumerate(new_predictions, start=1):
        print(f"Candidate {i}: ₹{int(round(val)):,}")
    '''
    print(y.head())
    print("="*50)
    print("Random Forest")
    print("="*50)
    
    
    rf=RandomForestRegressor(n_estimators=100, random_state=42)
    
    rf.fit(X_train,y_train)
    
    y_pred=rf.predict(X_test)
    
    print("Y_pred",y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='crimson')
    plt.xlabel("Actual CTC")
    plt.ylabel("Predicted CTC")
    plt.title("Actual vs Predicted CTC")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Train score is",rf.score(X_train,y_train))
    print("Test score is",rf.score(X_test,y_test))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    
    print("Test some data from x_test")
    
    # New candidate inputs
    new_input = pd.DataFrame([
        [1, 1, 'NA', 'NA', 'NA', 'NA', 'NA', 'PG', 'Arts', 'Lucknow', 2020, None, None, None, None, None,
        'Guwahati', 'Pune', 1, 'N', 'NA', 1, 1, 1, 1],
        [23, 14, 'HR', 'Consultant', 'Analytics', 'H', 'HR', 'Doctorate', 'Chemistry', 'Surat', 1988, 'Others',
        'Surat', 1990, 'Chemistry', 'Mangalore', 1997, 'Bangalore', 'Nagpur', 2702664, 'Y',
        'Key_Performer', 2, 4, 1, 1]
    ], columns=X.columns)

    # Fill missing values
    for col in new_input.columns:
        if new_input[col].dtype == 'object':
            new_input[col] = new_input[col].fillna('None')
        else:
            new_input[col] = new_input[col].fillna(0)

    # Encode and scale
    new_encoded = pd.get_dummies(new_input)
    new_encoded = new_encoded.reindex(columns=X_encoded.columns, fill_value=0)
    new_encoded[numerical] = scaler.transform(new_encoded[numerical])

    # Predict
    predictions = rf.predict(new_encoded)
    for i, val in enumerate(predictions, start=1):
        print(f"Candidate {i} expected CTC: ₹{int(round(val)):,}")

        

except Exception as e:
    print("\nAn error occurred:", e)
