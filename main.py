# Przygotowanie danych
import data_preparation
# import spreparowanych danych
df = data_preparation.data_prep(True, True)

print("\n Dane, na bazie których konstruujemy model:")
print(df[0])
print("\n Dane, gdzie przewidujemy pm2.5:")
print(df[1])
