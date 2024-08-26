import pandas as pd

# Wczytanie danych z pliku CSV
df = pd.read_csv('out_ciag_swiatel.csv')

# Funkcja do znalezienia i oznaczenia liczby powtórzeń sekwencji
def mark_repeated_sequences(dataframe):
    sequences = []
    sequence_counts = {}
    
    # Przejdź przez dane, sprawdzając co 3 linie
    for i in range(0, len(dataframe) - 2, 3):
        if i + 2 < len(dataframe):
            # Wyciągnij sekwencję 3 linii jako tuple
            sequence = tuple(dataframe.iloc[i:i+3].values.flatten())
            sequences.append((sequence, i))
    
    # Zliczanie powtórzeń sekwencji
    for seq, idx in sequences:
        if seq in sequence_counts:
            sequence_counts[seq].append(idx)
        else:
            sequence_counts[seq] = [idx]
    
    # Dodaj nową kolumnę do dataframe, oznaczającą liczbę powtórzeń dla zduplikowanych linii
    dataframe['sequence_count'] = 0
    for seq, indices in sequence_counts.items():
        if len(indices) > 1:
            for idx in indices:
                dataframe.loc[idx:idx+2, 'sequence_count'] = len(indices)
    
    return dataframe

# Wywołanie funkcji i zapisanie wyników
marked_df = mark_repeated_sequences(df)

# Wyświetlenie dataframe z zaznaczonymi powtórzeniami dla każdej sekwencji
print(marked_df)
