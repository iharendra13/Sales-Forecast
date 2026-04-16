import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv("sales.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna()

df['year']=df['Date'].dt.year
df['month']=df['Date'].dt.month
df['day']=df['Date'].dt.day
df['week']=df['Date'].dt.isocalendar().week.astype(int)

df['lag_1']=df['Sales'].shift(1)
df['lag_7']=df['Sales'].shift(7)
df=df.dropna()

X=df[['year','month','day','week','lag_1','lag_7']]
y=df['Sales']

model=RandomForestRegressor()
model.fit(X,y)

pickle.dump(model,open("model.pkl","wb"))
print("Model ready ✅")