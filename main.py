import os
import sys
import uvicorn

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    try:
        from server import app
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)
    except ImportError as e:
        print(f"Import error: {e}")
        print("Available files in src:", os.listdir('src'))
        sys.exit(1)