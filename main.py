from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

model = joblib.load('Telco_customer_churn_new.pkl')

class Churn_details(BaseModel):
    gender : Literal['Male','Female']
    SeniorCitizen : Literal['Yes','NO']
    Partner : Literal['Yes','No']
    Dependents : Literal['Yes','No']
    tenure : int = Field(...,ge=1,le=50)
    PhoneService : Literal['Yes','No']
    MultipleLines : Literal['Yes','No']
    InternetService : Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity : Literal['Yes','No']
    OnlineBackup : Literal['Yes','No']
    DeviceProtection : Literal['Yes','No']
    TechSupport : Literal['Yes','No']
    StreamingTV : Literal['Yes','No']
    StreamingMovies : Literal['Yes','No']
    Contract : Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling : Literal['Yes','No']
    PaymentMethod : Literal['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
    MonthlyCharges : float = Field(...,ge=0,le=100)
    TotalCharges : float = Field(...,ge=0,le=10000)

class predictionResponse(BaseModel):
    churn : str

@app.get('/')
def greet():
    return('Welcome to Customer churn prediction')


@app.post('/predict')
def predict(data : Churn_details):
    
    input_df = pd.DataFrame([{
        'gender' : data.gender,
        'SeniorCitizen' : data.SeniorCitizen,
        'Partner' : data.Partner,
        'Dependents' : data.Dependents,
        'tenure': data.tenure,
        'PhoneService' : data.PhoneService,
        'MultipleLines': data.MultipleLines,
        'InternetService' : data.InternetService,
        'OnlineSecurity' : data.OnlineSecurity,
        'OnlineBackup' : data.OnlineBackup,
        'DeviceProtection': data.DeviceProtection,
        'TechSupport' : data.TechSupport,
        'StreamingTV' : data.StreamingTV,
        'StreamingMovies' : data.StreamingMovies,
        'Contract' : data.Contract,
        'PaperlessBilling' : data.PaperlessBilling,
        'PaymentMethod' : data.PaymentMethod,
        'MonthlyCharges' : data.MonthlyCharges,
        'TotalCharges' : data.TotalCharges,
    }])

    prediction = model.predict(input_df)[0]
    result = "Yes" if prediction == 1 else "No"
    return predictionResponse(churn=result)