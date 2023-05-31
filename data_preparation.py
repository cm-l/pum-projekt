# do dataframe'ów
import pandas as pd

# do konwertowania daty na dzień tygodnia
from datetime import datetime


def data_prep(dropDay, dropEmpty):
    # Wczytujemy dane
    df = pd.read_csv('prsa_data.csv')

    # Czyszczenie i przygotowanie
    # usuwamy kolumnę z numerem wiersza
    df = df.drop('No', axis=1)
    # kierunek wiatru będzie reprezentować int
    # każdy z kierunków na róży wiatrów to osobny int
    # cv (calm and variable) to 0
    direction_mapping = {
        'NW': 1,
        'NE': 2,
        'SW': 3,  # wiatr z tej strony nie wieje w zbiorze danych
        'SE': 4,
        'cv': 0
    }

    # mapowanie kierunków
    df['cbwd'] = df['cbwd'].map(direction_mapping)

    # dodawanie dnia tygodnia
    # utworzenie kolumny z datą w formacie datetime
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    # wyciągnięcie z daty dnia tygodnia
    df['day_of_week'] = df['date'].dt.day_name()
    # dzień tygodnia w formacie: poniedziałek = 1, wtorek = 2, ..., niedziela = 7
    df['day_of_week'] = df['date'].dt.weekday + 1

    # usunięcie kolumny z datą (nie jest już potrzebna)
    df = df.drop('date', axis=1)
    # usunięcie kolumny z dniem? może nie być relewantny do modelu (w przeciwieństwie do dnia tygodnia)?
    if dropDay:
        print("Usuwanie kolumny z dniem miesiąca.")
        df = df.drop('day', axis=1)

    if dropEmpty:
        # sprawdzamy i usuwamy puste komórki
        to_delete = []  # lista z wierszami, które usuniemy przez wybrakowane dane
        print("Usuwanie wierszów z brakującymi danymi:")
        for index, row in df.iterrows():
            # brakująca wartość
            missing_columns = row[row.isna()].index.tolist()
            # brakujące pm2.5 (y) jest ok
            if "pm2.5" in missing_columns:
                missing_columns.remove("pm2.5")

            if len(missing_columns) > 0:
                print(f"W wierszu o indeksie {index} brakuje wartości w kolumnach: {', '.join(missing_columns)}")
                to_delete.append(index)

        # Remove the rows from the DataFrame
        df = df.drop(to_delete)

        # Reset the index of the DataFrame
        df.reset_index(drop=True)
        print("Usunięto wiersze z niepoprawnymi danymi.\n")

    # oddzielamy komórki z pustym pm2.5 do innego dataframe
    pm25_df = df[df['pm2.5'].isna()]
    df = df.dropna(subset=['pm2.5'])

    # zwróć zbiór danych po przygotowaniu
    # pierwszy to dane do modelu, drugi dla estymowanych wierszy
    return df, pm25_df
