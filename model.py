import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = pd.read_csv('/data/Iris.csv', header=None)
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

iris.info()

iris.head()

X = iris.drop('class', axis=1)
y = iris['class']

X.head()
y.tail()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy}')

