import subprocess
import os

# Ścieżki do pliku konfiguracyjnego i pliku wyjściowego
config_file = r"c:\DATA\ROB\7SEM\GIT\test\skrzyzowanieX\01_skrzyzowanieX.sumocfg"
output_file = r"c:\DATA\ROB\7SEM\GIT\test\python\out_tripInfo.xml"  # Dodaj rozszerzenie .xml

# Komenda do uruchomienia symulacji SUMO
command = [
    "sumo",  # Użyj "sumo" jeśli chcesz uruchomić bez GUI
    "-c", config_file,
    "--tripinfo-output", output_file
]

try:
    # Uruchomienie komendy
    result = subprocess.run(command, capture_output=True, text=True)

    # Wyświetlenie wyników
    print("Return code:", result.returncode)
    print("Standard output:", result.stdout)
    print("Standard error:", result.stderr)

    # Sprawdzenie, czy plik wyjściowy został utworzony
    if os.path.exists(output_file):
        print(f"Output file '{output_file}' has been created successfully.")
    else:
        print(f"Output file '{output_file}' was not created.")

except FileNotFoundError as e:
    print(f"Error: {e.strerror}. The command or file might be incorrect.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
