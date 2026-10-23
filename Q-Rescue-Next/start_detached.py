"""
start_detached.py - Launches Streamlit as a persistent Windows background process using CREATE_NO_WINDOW and Python UTF-8 mode.
"""

import os
import subprocess
import sys
import time

def main():
    base_dir = r"D:\PROJECTS\Q-Rescue-Next"
    python_exe = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
    app_py = os.path.join(base_dir, "app.py")
    log_file = os.path.join(base_dir, "streamlit_server.log")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [
        python_exe,
        "-X",
        "utf8",
        "-m",
        "streamlit",
        "run",
        app_py,
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
    ]

    with open(log_file, "a", encoding="utf-8") as out:
        out.write(f"\n--- Streamlit Launched at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        out.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=out,
            cwd=base_dir,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    print(f"Streamlit running with PID {proc.pid}")

if __name__ == "__main__":
    main()
