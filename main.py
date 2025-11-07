# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from dotenv import load_dotenv
# import uvicorn
# import joblib
# import os
# import numpy as np




# load_dotenv()

# app = FastAPI()

# model = joblib.load("model.pkl")

# scaler = joblib.load("scaler.pkl")


# class WineFeatures(BaseModel):
#     fixed_acidity : float
#     volatile_acidity : float
#     citric_acid : float
#     residual_sugar : float
#     chloride : float
#     free_sulfur_dioxide : float
#     total_sulfur_dioxide : float
#     density : float
#     ph : float
#     sulphate : float
#     alcohol : float




# # Let's define our get endpoint to display a welcome message

# @app.get("/")
# def welcome_page():
#     return {
#         "message": "Welcome to this fabulous homepage, thank you for coming."
#     }


# @app.post("/Predict")
# def WinePredict(wine:WineFeatures):
#     try:
#         features = np.array([[
#         wine.fixed_acidity,
#         wine.volatile_acidity,
#         wine.citric_acid,
#         wine.residual_sugar,
#         wine.chloride,
#         wine.free_sulfur_dioxide,
#         wine.total_sulfur_dioxide,
#         wine.density,
#         wine.ph,
#         wine.sulphate,
#         wine.alcohol

#         ]])

#         scaled_features = scaler.transform(features)

#         predictions = model.predict(scaled_features)

#         return {
#         "Predicted Quality": str(predictions[0])
#         }

    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))




# if __name__ == "__main__":
#     uvicorn.run(app,host = os.getenv("host"),port = int(os.getenv("port")))



















from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
import os
from dotenv import load_dotenv


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report


from random import randint

load_dotenv()

# Initializing the application
app = FastAPI()

# Loading best model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Creating the


class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxides: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


# Building endpoint
@app.get("/")
def home():
    return {"Message": "Welcome to Wine quality predictor Model"}


@app.post("/predict")
def get_wine_quality(input: WineFeatures):
    features = np.array([[
        input.fixed_acidity,
        input.volatile_acidity,
        input.citric_acid,
        input.residual_sugar,
        input.chlorides,
        input.free_sulfur_dioxide,
        input.total_sulfur_dioxides,
        input.density,
        input.pH,
        input.sulphates,
        input.alcohol
    ]])

    scaled_feature = scaler.transform(features)
    prediction = model.predict(scaled_feature)

    return {"Predicted wine quality": str(prediction[0])}


uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("host"), port=int(os.getenv("port")))