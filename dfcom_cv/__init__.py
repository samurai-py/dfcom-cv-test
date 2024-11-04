import subprocess
import sys

def main():
    """Função para executar o aplicativo Streamlit."""
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dfcom_cv/app/app.py'])

if __name__ == "__main__":
    main()