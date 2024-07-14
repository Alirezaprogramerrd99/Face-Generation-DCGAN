# main.py
import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(filename='run_scripts.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Get the path to the current Python executable
python_executable = sys.executable
TRAIN_TEST_MODEL = True

# Check and create the output directory if it doesn't exist
output_dir = 'tests\\test_outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logging.info(f"Created directory: {output_dir}")

def run_scripts_sequentially():
    scripts = ["tests\\test_cpus.py", "tests\\test_model.py", "tests\\test_model_creation.py"]

    for script in scripts:
        result = subprocess.run([python_executable, script], capture_output=True, text=True)
        print(f"Running {script}:\n\n{result.stdout}")
        if result.stderr:
            print(f"Error in {script}:\n{result.stderr}")


def run_scripts_sequentially_write_to_file():
    scripts = ["tests\\test_cpus.py", "tests\\test_model.py", "tests\\test_model_creation.py"]

    for script in scripts:
        script_name = os.path.basename(script)
        output_file = os.path.join(output_dir, f"{script_name}.txt")
        
        try:
            with open(output_file, 'w') as file:
                result = subprocess.run([python_executable, script], capture_output=True, text=True)
                print(f"Running {script}")
                file.write(result.stdout)
                logging.info(f"Running {script}:\n\n{result.stdout}")

                if result.stderr:
                    file.write(f"Error in {script}:\n{result.stderr}")
                    logging.error(f"Error in {script}: {result.stderr}")
                else:
                    logging.info(f"Successfully ran {script}")

            print(f"Output written to {output_file}")
            
        except Exception as e:   # it doesn't halt the execution of the whole project...
            logging.error(f"Failed to run {script}: {e}")
            print(f"Failed to run {script}. Check the log file for details.")

if __name__ == "__main__":

    logging.info("Script execution started")
    print("python_executable:\n", python_executable)
    run_scripts_sequentially_write_to_file()
    logging.info("the main script execution completed")