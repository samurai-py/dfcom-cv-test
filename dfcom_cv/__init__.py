import subprocess
import sys

def run_streamlit_app():
    """Função para executar o aplicativo Streamlit."""
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app/app.py'])

if __name__ == "__main__":
    run_streamlit_app()
