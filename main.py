# Przygotowanie danych
from sklearn.ensemble import RandomForestRegressor

import data_preparation
# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# import spreparowanych danych
df = data_preparation.data_prep(False, True)

print("\n Dane, na bazie których konstruujemy model:")
print(df[0])
print("\n Dane, gdzie przewidujemy pm2.5:")
print(df[1])


df_main = df[0]
df_to_predict = df[1]

# obserwacje a wartości przewidywane
X = df_main.drop(columns=['pm2.5'])
y = df_main['pm2.5']

# podział na zbiór testowy i treningowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=707)

# # 1. REGRESJA LINIOWA
# model = LinearRegression()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# # Ocena modelu regresji liniowej
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("\n1. Regresja liniowa:")
# print("MSE:", mse)
# print("R^2:", r2)


# 1. REGRESJA LASSO
model_lasso = Lasso(alpha=0.02, random_state=4664464)  # alpha do dostosowania
model_lasso.fit(X_train, y_train)

y_pred_lasso = model_lasso.predict(X_test)

# Ocena modelu
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\n1. Regresja LASSO")
print("MSE:", mse_lasso)
print("R^2 dla testowych:", r2_lasso)
y_pred_lasso = model_lasso.predict(X_train)
print("R^2 dla treningowych: %.2f" % r2_score(y_train, y_pred_lasso))

# 2. RANDOM FOREST
model_forest = RandomForestRegressor(random_state=123456)
model_forest.fit(X_train, y_train)

y_pred_forest = model_forest.predict(X_test)

# Ocena modelu
mse_forest = mean_squared_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)

print("\n2. Regresja Random Forest")
print("MSE:", mse_forest)
print("R^2 dla testowych:", r2_forest)
y_pred_forest = model_forest.predict(X_train)
print("R^2 dla treningowych: %.2f" % r2_score(y_train, y_pred_forest))


# 2. REGRESJA KNN


from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors

#Normalizacja danych
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#Szukanie najlepszego k

print("\n Szukanie najlepszego K")

rmse_val = [] #to store rmse values for different k
for K in range(10):
    K = K+1
    knn = neighbors.KNeighborsRegressor(n_neighbors = K)

    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    print('R^2 for k= ' , K , 'is:', r2_score(y_test, y_pred_knn))

#Najlepszy k jest 1


knn = neighbors.KNeighborsRegressor(n_neighbors = 1)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Ocena modelu

print("\n3. Regresja KNN")
print("MSE: %.2f" % mean_squared_error(y_test, y_pred_knn))
# The coefficient of determination: 1 is perfect prediction
print("R^2 dla testowych: %.2f" % r2_score(y_test, y_pred_knn))
y_pred_knn = knn.predict(X_train_scaled)

print("R^2 dla treningowych: %.2f" % r2_score(y_train, y_pred_knn))

# Przewidywanie brakujących danych na bazie modelu lasso
lasso_predictions = model_lasso.predict(df_to_predict.drop(columns=['pm2.5']))
df_to_predict['pm2.5'] = lasso_predictions

print(df_to_predict)

# Przewidywanie brakujących danych na bazie modelu random forest
forest_predictions = model_forest.predict(df_to_predict.drop(columns=['pm2.5']))
df_to_predict['pm2.5'] = forest_predictions

print(df_to_predict)

# # Przewidywanie brakujących danych na bazie modelu knn (ten daje dziwne wyniki, warto to skonsultować na zajęciach)
# knn_predictions = knn.predict(df_to_predict.drop(columns=['pm2.5']))
# df_to_predict['pm2.5'] = knn_predictions

# print(df_to_predict)

