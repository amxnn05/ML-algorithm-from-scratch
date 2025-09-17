import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text  import CountVectorizer


df = pd.read_csv("Dataset/spam_ham_dataset.csv")
df= df.drop(['Unnamed: 0'], axis=1)
x = df['text'].values
y = df['label_num'].values

cv = CountVectorizer()

x = cv.fit_transform(x)

bnb = BernoulliNB(binarize=0.0)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=42 )


bnb.fit(x_train, y_train)
y_pred = bnb.predict(x_test)

print(classification_report(y_test, y_pred))