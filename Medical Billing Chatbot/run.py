import subprocess
import os
import sys
import time
import webbrowser
from threading import Thread

def run_backend():
    """Run the FastAPI backend server"""
    print("Starting FastAPI backend server...")
    subprocess.run(["uvicorn", "backend:app" , "--host", "0.0.0.0", "--port", "8000", "--reload"], check=True)

def run_frontend():
    """Run the Streamlit frontend app"""
    print("Starting Streamlit frontend...")
    subprocess.run(["streamlit", "run", "app.py" , "--server.headless=true"] , check=True)

def open_browser():
    """Open browser tabs for the app and API docs after a delay"""
    time.sleep(3) 
    
    webbrowser.open("http://localhost:8501")
    
    webbrowser.open("http://localhost:8000/docs")

if __name__ == "__main__":

    required_files = ["app.py", "backend.py", "connect_memory.py", "create_memory.py"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: Required file {file} not found!")
            sys.exit(1)
    
    backend_thread = Thread(target=run_backend)
    frontend_thread = Thread(target=run_frontend)
    browser_thread = Thread(target=open_browser)
    
    backend_thread.start()
    time.sleep(2)
    frontend_thread.start()
    browser_thread.start()
    

    try:
        backend_thread.join()
        frontend_thread.join()
    except KeyboardInterrupt:
        print("\nShutting down application...")
        sys.exit(0)