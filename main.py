# Przygotowanie danych
from sklearn.ensemble import RandomForestRegressor

import data_preparation
# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
# import spreparowanych danych
df = data_preparation.data_prep(True, True)

print("\n Dane, na bazie których konstruujemy model:")
print(df[0])
print("\n Dane, gdzie przewidujemy pm2.5:")
print(df[1])

df_main = df[0]

# obserwacje a wartości przewidywane
X = df_main.drop(columns=['pm2.5'])
y = df_main['pm2.5']

# podział na zbiór testowy i treningowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=707)

# 1. REGRESJA LINIOWA
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Ocena modelu regresji liniowej
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n1. Regresja liniowa:")
print("MSE:", mse)
print("R^2:", r2)

# 2. RANDOM FOREST
model_forest = RandomForestRegressor(random_state=123456)
model_forest.fit(X_train, y_train)

y_pred_forest = model_forest.predict(X_test)

# Ocena modelu
mse_forest = mean_squared_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)

print("\n2. Regresja Random Forest")
print("MSE:", mse_forest)
print("R^2:", r2_forest)

# 3. REGRESJA LASSO
model_lasso = Lasso(alpha=0.02, random_state=4664464)  # alpha do dostosowania
model_lasso.fit(X_train, y_train)

y_pred_lasso = model_lasso.predict(X_test)

# Ocena modelu
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\n2. Regresja LASSO")
print("MSE:", mse_lasso)
print("R^2:", r2_lasso)