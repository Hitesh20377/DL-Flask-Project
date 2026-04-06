import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# data load
data = pd.read_csv("data.csv")

# features & target
X = data.drop("price", axis=1)   # apne target ka naam check kar
y = data["price"]

# model train
model = LinearRegression()
model.fit(X, y)

# save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved as model.pkl")