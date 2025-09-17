import numpy as np
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score




class gaussian_nb:
    def __init__(self):
        pass
    
             








# importing the dataset 
df = pd.read_csv("Dataset/loan_approval_dataset.csv")
df.head()
df.sample(5)
df.columns = df.columns.str.strip()
# print(df.columns.tolist())

X = df[['education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score']].copy()
y = df['loan_status'].copy()
