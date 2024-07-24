import subprocess
import os

# Ścieżki do pliku wejściowego i wyjściowego
tripinfo_file = r"c:\DATA\ROB\7SEM\GIT\test\python\out_tripInfo.xml"
output_file = r"c:\DATA\ROB\7SEM\GIT\test\python\trip_statistics_output.xml"

# Komenda do uruchomienia skryptu tripStatistics.py
command = [
    "python", r"C:\Program Files (x86)\Eclipse\Sumo\tools\output\tripStatistics.py",
    "-t", tripinfo_file,
    "-o", output_file

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
