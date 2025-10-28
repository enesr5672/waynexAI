import numpy as np
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def home():
    return {'message': 'WAYNEX AI System Running'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
