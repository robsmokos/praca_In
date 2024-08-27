import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations

def calculate_similarity(seq1, seq2, threshold=0.1):
    diffs = [abs(a - b) / max(abs(a), abs(b), 1) for a, b in zip(seq1, seq2)]
    return all(diff <= threshold for diff in diffs)

def process_file(file_path):
    # Wczytanie danych z pliku CSV
    df = pd.read_csv(file_path)
    
    # Ekstrakcja sekwencji 6 kolejnych wartości w kolumnie 'time_on_duration'
    sequences_duration = [
        tuple(df['time_on_duration'].iloc[i:i+6])
        for i in range(len(df) - 5)
    ]
    
    # Filtrowanie sekwencji, które zawierają 0 lub NaN
    sequences_filtered = [
        seq for seq in sequences_duration
        if not any(np.isnan(value) or value == 0.0 for value in seq)
    ]
    
    # Liczenie wystąpień sekwencji bez użycia diff
    sequence_counts = Counter(sequences_filtered)
    
    # Wyświetlanie 3 najczęstszych sekwencji bez użycia diff
    print("Najczęstsze sekwencje bez użycia diff:")
    for sequence, count in sequence_counts.most_common(3):
        print(f"{sequence} - Występuje {count} razy")
    
    # Grupowanie podobnych sekwencji na podstawie progu podobieństwa (diff)
    similar_sequences_counts = Counter()
    
    for seq1, seq2 in combinations(sequences_filtered, 2):
        if calculate_similarity(seq1, seq2, threshold=0.1):
            similar_sequences_counts[seq1] += 1
            similar_sequences_counts[seq2] += 1
    
    # Wyświetlanie 3 najczęstszych podobnych sekwencji z użyciem diff
    print("\nNajczęstsze podobne sekwencje z użyciem diff:")
    for sequence, count in similar_sequences_counts.most_common(3):
        print(f"{sequence} - Występuje {count} razy")

# Ścieżka do pliku
file_path = 'a2_out_out_jak_dlugo_wlaczone.csv'

# Przetwarzanie pliku
process_file(file_path)
