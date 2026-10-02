import uvicorn
from src.api.server import app

def main():
    print("Starting FTMM COMPASS Chatbot Backend on http://0.0.0.0:8000 ...")
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
