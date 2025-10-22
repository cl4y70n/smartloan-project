from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

@app.get('/')
def root():
    return {'message': 'SmartLoan API funcionando!'}
