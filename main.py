from classes import Grid, ArcTask
from utils import load_all, load_arc1, load_arc2
from api import app
import uvicorn

def main():
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()