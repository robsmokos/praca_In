import subprocess

script_path = r"C:\Program Files (x86)\Eclipse\Sumo\tools\output\tripStatistics.py"
input_file = r"c:\DATA\ROB\7SEM\GIT\test\python\tripinfo.xml"
output_file = r"c:\DATA\ROB\7SEM\GIT\test\python\trip_statistics_output.xml"

command = [
    "python", script_path,
    "-t", input_file,
    "-o", output_file
]

result = subprocess.run(command, capture_output=True, text=True)

print("Return code:", result.returncode)
print("Standard output:", result.stdout)
print("Standard error:", result.stderr)