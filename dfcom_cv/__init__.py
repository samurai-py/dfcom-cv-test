import subprocess
import sys

def main():
    """Function to run the Streamlit application."""
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dfcom_cv/app/app.py'])

if __name__ == "__main__":
    main()
