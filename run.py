import subprocess
import os

venv_python = os.path.join(".env", "Scripts", "python.exe")  # for Windows

scripts = ["search_news.py", "create_features.py", "train_model.py"]

for script in scripts:
    try:
        print(f"Executing {script} with Python from virtual .env...")
        subprocess.run([venv_python, script], check=True)  
    except subprocess.CalledProcessError as e:
        print(f"Execution error {script}: {e}")
        break  

