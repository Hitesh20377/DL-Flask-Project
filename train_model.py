import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = pd.read_csv("data.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

y_max = y.max()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train / y_max
y_test = y_test / y_max

model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, verbose=1)

model.save("model.keras")

pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(y_max, open("y_max.pkl", "wb"))

print("Model trained & saved successfully")