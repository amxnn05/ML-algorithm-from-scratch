import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer

warnings.filterwarnings('ignore')

df = pd.read_csv('dataset/Position_Salaries.csv')
df.sample(5)
df

X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

le = LabelEncoder()

X_cat = df.select_dtypes('object').apply(le.fit_transform)
x_num = df.select_dtypes(exclude='object').values

x = pd.concat([pd.DataFrame(x_num), X_cat], axis=1).values

regressor = RandomForestRegressor(n_estimators=10, random_state=0)

regressor.fit(x,y)

y_pred = regressor.predict(x)
print(r2_score(y, y_pred))
print(mean_squared_error(y, y_pred))