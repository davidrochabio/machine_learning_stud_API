# Data/Path Handling
import logging
import pandas as pd
import numpy as np
import pickle
import json
import os
from pydantic import BaseModel, create_model

# Server
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

# Initialize logging
my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG, filename='sample.log')

# Get Pickle path
path = os.path.dirname(os.path.abspath(__file__)).replace('app', 'app/data')

# Initialize files
model = pickle.load(open(path + '/model.pickle', 'rb'))
features = pickle.load(open(path + '/features.pickle', 'rb'))
model_eval_metrics = pickle.load(open(path + '/metrics.pickle', 'rb'))

# Instatiates the class (BaseModel)
class Data(BaseModel):
    age: int
    Medu: int
    Fedu: int
    traveltime: int
    studytime: int
    failures: int
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    abscences: int
    G1: int
    G2: int
    

# post to get predictions
@app.post("/predict/")
async def predict(data: Data):
    try:
        # Extract data in correct order
        data_dict = jsonable_encoder(data)
        for key, value in data_dict.items():
            data_dict[key] = [value]
            to_predict = pd.DataFrame.from_dict(data_dict)
    
    except:
        my_logger.error("Something went wrong!")
        return {"prediction": "error"}
        
    # Create and return prediction
    prediction = model.predict(to_predict)
    return {"prediction": float(prediction[0])}

# get to retrieve the model's metrics
@app.get("/metrics/")
async def metrics():
    return {"model_metrics": model_eval_metrics}