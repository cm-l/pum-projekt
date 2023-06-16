# Przygotowanie danych
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import data_preparation
# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model

# ładne grafiki
import seaborn as sns

# import spreparowanych danych
df = data_preparation.data_prep(False, True)

print("\n Dane, na bazie których konstruujemy model:")
print(df[0])
print("\n Dane, gdzie przewidujemy pm2.5:")
print(df[1])

#korelacja
print("Korelacje:")
print(df[0].corr()["pm2.5"])
# Wykres
correlations = df[0].corr()["pm2.5"]
corr_matrix = correlations.to_frame()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=False)
plt.title("Macierz korelacji")
plt.show()

#Pomiedzy pm2.5 i IWS jest najwieksza korelacja, wiec wykresy beda dla zaleznosci pm2.5 od IWS dla lepszego zobrazowania, do ustalenia czy jest sens dropnac kolumny

#Rozklad pm2.5
#plt.hist(df[0]["pm2.5"])
#plt.show()

# potem to zakomentuję bo mnie coś strzeli jak się powoli włączają w pycharmie te wykresy
# połowa czasu uruchamiania kodu to czekanie na wykres - super community edition

# WYKRES JEDEN
# PM2.5 w zależności od czasu
# to już powinno być w dataprepie ale na wszelki wypadek
df_filtered = df[0].dropna(subset=['pm2.5'])

pm25_values = df_filtered['pm2.5']
dates = pd.to_datetime(df_filtered[['year', 'month', 'day', 'hour']])

plt.figure(figsize=(12, 6))
plt.plot(dates, pm25_values, color='red')

plt.title('Stężenie PM2.5 w powietrzu w obserwowanym okresie')
plt.xlabel('Data')
plt.ylabel('PM2.5')

plt.xticks(rotation=45)
plt.show()


#Zaleznosc pm2.5 od Iws
#plt.scatter(df[0]["Iws"],df[0]["pm2.5"])
#plt.show()

# to już powinno być w dataprepie ale na wszelki wypadek
df_filtered = df[0].dropna(subset=['Iws', 'pm2.5'])

iws_values = df_filtered['Iws']
pm25_values = df_filtered['pm2.5']

plt.figure(figsize=(8, 6))
plt.scatter(iws_values, pm25_values, color='grey', alpha=0.37)

plt.title('Zależność między prędkością wiatru a PM2.5')
plt.xlabel('Prędkość wiatru')
plt.ylabel('PM2.5')

plt.show()


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
#do wykresu potem
lasso_predictions=y_pred_lasso
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
#do wykresu potem
forest_predictions=y_pred_forest
# Ocena modelu
mse_forest = mean_squared_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)

print("\n2. Regresja Random Forest")
print("MSE:", mse_forest)
print("R^2 dla testowych:", r2_forest)
y_pred_forest = model_forest.predict(X_train)
print("R^2 dla treningowych: %.2f" % r2_score(y_train, y_pred_forest))


# # 3. REGRESJA KNN


# from sklearn.preprocessing import MinMaxScaler
# from sklearn import neighbors

# #Normalizacja danych
# scaler = MinMaxScaler(feature_range=(0, 1))
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.fit_transform(X_test)

# #Szukanie najlepszego k

# print("\n Szukanie najlepszego K")

# rmse_val = [] #to store rmse values for different k
# for K in range(10):
#     K = K+1
#     knn = neighbors.KNeighborsRegressor(n_neighbors = K)

#     knn.fit(X_train_scaled, y_train)
#     y_pred_knn = knn.predict(X_test_scaled)
#     print('R^2 for k= ' , K , 'is:', r2_score(y_test, y_pred_knn))

# #Najlepszy k jest 1


# knn = neighbors.KNeighborsRegressor(n_neighbors = 1)
# knn.fit(X_train_scaled, y_train)
# y_pred_knn = knn.predict(X_test_scaled)

# # Ocena modelu

# print("\n3. Regresja KNN")
# print("MSE: %.2f" % mean_squared_error(y_test, y_pred_knn))
# # The coefficient of determination: 1 is perfect prediction
# print("R^2 dla testowych: %.2f" % r2_score(y_test, y_pred_knn))
# y_pred_knn = knn.predict(X_train_scaled)

# print("R^2 dla treningowych: %.2f" % r2_score(y_train, y_pred_knn))


# 3 Model Wielomianowy
poly = PolynomialFeatures(degree=3, include_bias=False)
# Degree 3 daje najlepsze wyniki dla testowych, wiecej i mamy overfitting
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)


poly_reg_model = linear_model.LinearRegression()
poly_reg_model.fit(X_train_poly,y_train)

Y_pred_poly = poly_reg_model.predict(X_test_poly)
#do wykresu potem
poly_predictions = Y_pred_poly

#Dane regresji wielomianowej
print("\n3. Regresja wielomianowa")
# The mean squared error
print("MSE: %.2f" % mean_squared_error(y_test, Y_pred_poly))
# The coefficient of determination: 1 is perfect prediction
print("R^2 dla testowych: %.2f" % r2_score(y_test, Y_pred_poly))
Y_pred_poly = poly_reg_model.predict(X_train_poly)

print("R^2 dla treningowych: %.2f" % r2_score(y_train, Y_pred_poly))

#Dla dodatkowych wykresów zobaczymy czy modele tworza podobne do rzeczywistych pm2.5 dla danych testowych

df_lasso = X_test
df_random_forest = X_test
df_poly = X_test
#Dla lasso
df_lasso['pm2.5'] = lasso_predictions

#Rozklad pm2.5
plt.hist(df_lasso["pm2.5"])
plt.show() 
#Zaleznosc pm2.5 od Iws
plt.scatter(df_lasso["Iws"],df_lasso["pm2.5"])
plt.show()

# Real
plt.subplot(1, 2, 1)
plt.scatter(df[0]['Iws'], df[0]['pm2.5'], color='blue', alpha=0.5)
plt.xlabel('Prędkość wiatru')
plt.ylabel('PM2.5')
plt.title('Prawdziwe dane')

# LASSO - porównanie
plt.subplot(1, 2, 2)
plt.scatter(df_lasso['Iws'], df_lasso['pm2.5'], color='red', alpha=0.5)
plt.xlabel('Prędkość wiatru')
plt.ylabel('PM2.5')
plt.title('Model LASSO')

plt.tight_layout()
plt.show()

# OVERLAY DLA LASSO
plt.scatter(df[0]['Iws'], df[0]['pm2.5'], color='blue', alpha=0.5, label='Prawdziwe dane')
plt.scatter(df_lasso['Iws'], df_lasso['pm2.5'], color='red', alpha=0.5, label='Model LASSO')

plt.xlabel('Prędkość wiatru')
plt.ylabel('PM2.5')
plt.title('Prawdziwe dane vs. Model LASSO')

plt.legend()

plt.show()

#Ocena modelu: prosty w implementacji, ale daje beznadziejne rezultaty, słaby R^2 i rozklad danych kompletnie nie odpowiada rzeczywistosci w dodatku generuje ujemne pm2.5, ewidentnie brak relacji liniowej


#Dla random_forest
df_random_forest['pm2.5'] = forest_predictions

#Rozklad pm2.5
plt.hist(df_random_forest["pm2.5"])
plt.show() 
#Zaleznosc pm2.5 od Iws
plt.scatter(df_random_forest["Iws"],df_random_forest["pm2.5"])
plt.show()

# OVERLAY DLA RANDOM FORESTA
plt.scatter(df[0]['Iws'], df[0]['pm2.5'], color='blue', alpha=0.5, label='Prawdziwe dane')
plt.scatter(df_random_forest['Iws'], df_random_forest['pm2.5'], color='red', alpha=0.5, label='Model Random Forest')

plt.xlabel('Prędkość wiatru')
plt.ylabel('PM2.5')
plt.title('Prawdziwe dane vs. Model Random Forest')

plt.legend()

plt.show()

#ocena model: długi w nauce, ale daje najlepsze rezultaty, dobry R^2 i zwraca realne nieujemne pm2.5


#Dla wielomianowej
df_poly['pm2.5'] = poly_predictions

#Rozklad pm2.5
plt.hist(df_poly["pm2.5"])
plt.show() 
#Zaleznosc pm2.5 od Iws
plt.scatter(df_poly["Iws"],df_poly["pm2.5"])
plt.show() 

# OVERLAY DLA WIELOMIANOWEJ
plt.scatter(df[0]['Iws'], df[0]['pm2.5'], color='blue', alpha=0.5, label='Prawdziwe dane')
plt.scatter(df_poly['Iws'], df_poly['pm2.5'], color='red', alpha=0.5, label='Regresja wielomianowa')

plt.xlabel('Prędkość wiatru')
plt.ylabel('PM2.5')
plt.title('Prawdziwe dane vs. Model regresji wielomianowej')

plt.legend()

plt.show()

#ocena modelu: ok R^2, rozkład w miare podobny, chociaz wystepuja ujemne pm2.5, trudno merytorycznie uzasadnić zaleznosci na podstawie trzeciej potegi

# # Przewidywanie brakujących danych na bazie modelu lasso
# lasso_predictions = model_lasso.predict(df_to_predict.drop(columns=['pm2.5']))
# df_to_predict['pm2.5'] = lasso_predictions

# print(df_to_predict)

# # Przewidywanie brakujących danych na bazie modelu random forest
# forest_predictions = model_forest.predict(df_to_predict.drop(columns=['pm2.5']))
# df_to_predict['pm2.5'] = forest_predictions

# print(df_to_predict)

# # Przewidywanie brakujących danych na bazie modelu knn (ten daje dziwne wyniki, warto to skonsultować na zajęciach)
# knn_predictions = knn.predict(df_to_predict.drop(columns=['pm2.5']))
# df_to_predict['pm2.5'] = knn_predictions

# print(df_to_predict)




