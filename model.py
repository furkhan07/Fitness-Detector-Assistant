import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle


df = pd.read_csv('cleaned_fitness_data.csv')

# dropping unnamed columns
df.drop(columns=["Unnamed: 6", "Unnamed: 7", "Unnamed: 8", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11"], inplace=True)
df.head()

df.info()

# Assigning features and target variable
X = df.drop('BMI', axis=1)
y = df['BMI']

#Identifying numerical and categorical columns
num_cols = X.select_dtypes(include='number').columns
cat_cols = X.select_dtypes(include='object').columns

# Splitting the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, random_state=42)   

# Preprocessing using ColumnTransformer
compose= ColumnTransformer(
    transformers=[
        ('onehotencoder', OneHotEncoder(handle_unknown='ignore'), cat_cols),  # this is for Gender Column  Female - 0 Male - 1
    ],
    remainder='passthrough'  # this will leave the numerical columns
)

# Modeling using Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', compose),   
    ('model', XGBRegressor())  
])
model = XGBRegressor()
pipeline.fit(xtrain, ytrain)


#Training and Testing Score
pipeline.score(xtrain, ytrain)

pipeline.score(xtest, ytest)

# Saving the model using pickle
with open('fitness_bmi_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)   
print("Model saved successfully.") 
  