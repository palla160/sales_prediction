import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys
url="https://docs.google.com/spreadsheets/d/1XUF8drRl5ap1HjFLzgpYGD7qSqZGRdMp0U-jRN7p0rk/export?format=csv"
df=pd.read_csv(url)


df["time_index"]=df.groupby("product").cumcount()


le_product=LabelEncoder()
le_store=LabelEncoder()
df["product"]=le_product.fit_transform(df["product"])
df["store"]=le_store.fit_transform(df["store"])
x=df.drop("units_sold",axis=1)
y=df["units_sold"]

model=LinearRegression()
model.fit(x,y)

report = []

for product in df["product"].unique():

    subset = df[df["product"] == product]
    last_row = subset.iloc[-1]

    next_time = last_row["time_index"] + 1

    X_pred = [[
        last_row["product"],
        last_row["price"],
        last_row["store"],
        last_row["marketing_spend"],
        next_time
    ]]

    prediction = model.predict(X_pred)[0]

    decision = "Increase stock" if prediction > last_row["units_sold"] else "Maintain"

    name = le_product.inverse_transform([product])[0]

    report.append(f"{name}: Forecast {prediction:.0f} → {decision}")


final_report="\n".join(report)

with open("ml_report.txt","w",encoding="utf-8") as f:
    f.write(final_report)
print(final_report)

