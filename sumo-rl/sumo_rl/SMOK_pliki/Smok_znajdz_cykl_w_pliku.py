from collections import Counter
import re

def find_most_common_sequence(file_path):
    # Otwórz plik i wczytaj dane
    with open(file_path, 'r') as file:
        data = file.readlines()
    
    # Usuń znaki nowej linii i zbierz wszystkie liczby
    numbers = [line.strip() for line in data]
    
    # Zamień liczby na pojedyncze elementy
    numbers = [number for number in numbers if number]  # Usuwanie pustych linii
    
    # Wyszukaj wszystkie możliwe sekwencje o długości co najmniej 3
    sequences = []
    for start in range(len(numbers)):
        for length in range(4, len(numbers) - start + 1):
            sequence = tuple(numbers[start:start + length])
            sequences.append(sequence)
    
    # Policz częstość występowania każdej sekwencji
    counter = Counter(sequences)
    
    # Znajdź najczęściej występującą sekwencję
    if counter:
        most_common, count = counter.most_common(1)[0]
        most_common_str = ' '.join(most_common)
        print(f"Najczęściej występujący ciąg liczb o długości co najmniej 3 to: {most_common_str}")
        print(f"Ten cykl występuje {count} razy.")
    else:
        print("Nie znaleziono ciągu liczb o długości co najmniej 3")

# Podaj ścieżkę do pliku
file_path = 'CoIleZmiana2.csv'
find_most_common_sequence(file_path)
