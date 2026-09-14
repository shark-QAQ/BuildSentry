# start.py
"""
启动 BuildSentry 服务。
"""
import subprocess
import time
import sys

def run_backend():
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"])

def run_frontend():
    return subprocess.Popen([sys.executable, "ui/app.py"])

if __name__ == "__main__":
    print("[START] 启动 BuildSentry...")
    backend = run_backend()
    print(f"[OK] 后端 PID: {backend.pid}")
    time.sleep(3)
    frontend = run_frontend()
    print(f"[OK] 前端 PID: {frontend.pid}")
    print("[INFO] 访问 http://localhost:7860")
    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\n[STOP] 关闭...")
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()
        print("[OK] 已停止")
