import subprocess
import os

# Caminho para o interpretador Python dentro do ambiente virtual
venv_python = os.path.join(".env", "Scripts", "python.exe")  # Para Windows

# Lista de scripts a serem executados
scripts = ["search_news.py", "create_features.py", "train_model.py"]

for script in scripts:
    try:
        print(f"Executando {script} com o Python do ambiente virtual...")
        subprocess.run([venv_python, script], check=True)  # Executa com o Python do venv
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar {script}: {e}")
        break  # Interrompe a sequência em caso de erro

