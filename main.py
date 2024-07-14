# main.py
import subprocess
import sys

# Get the path to the current Python executable
python_executable = sys.executable
TRAIN_TEST_MODEL = True

def run_scripts_sequentially():
    scripts = ["tests\\test_cpus.py", "tests\\test_model.py"]

    for script in scripts:
        result = subprocess.run([python_executable, script], capture_output=True, text=True)
        print(f"Running {script}:\n\n{result.stdout}")
        if result.stderr:
            print(f"Error in {script}:\n{result.stderr}")

if __name__ == "__main__":
    print("python_executable:\n", python_executable)
    run_scripts_sequentially()