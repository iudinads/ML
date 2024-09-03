import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target


sns.pairplot(data, hue='species')
plt.show()

#Выбор признаков и целевой переменной
X = data[['petal length (cm)', 'petal width (cm)']]
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

#Оценка точности
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

def predict_species(petal_length, petal_width):
    prediction = model.predict(np.array([[petal_length, petal_width]]))
    return iris.target_names[np.round(prediction).astype(int)[0]]


predicted_species = predict_species(1.2, 2.0)
print(f'Predicted species: {predicted_species}')