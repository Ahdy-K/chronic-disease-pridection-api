
## Importing models
import diabetespredictionsystem
import cardiopathie
## FAST API
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "http://localhost:4200",
    "https://localhost:4200/doctors/ai/diabetes",
    "http://localhost:4200/doctors/ai/cardio",
    "http://localhost",
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SugarTestSample(BaseModel):
  Pregnancies:int
  Glucose: float
  BloodPressure: float
  SkinThickness: float
  Insulin: float
  BMI: float
  DiabetesPedigreeFunction: float
  Age: int

class HeartTestSample(BaseModel):
  age: int
  sex: int
  cp: float
  trestbps: float
  chol: float
  fbs: float
  restecg: float
  thalach: float
  exang: float
  oldpeak: float
  slope: float
  ca: float
  thal: float


@app.post("/diabetes")
def diabetes_predictor(s:SugarTestSample):
  json_sample = jsonable_encoder(s)
  sample_list = list(json_sample.values())
  print("SAMPLE DATA",sample_list)
  return diabetespredictionsystem.diabetes_pridection_model(sample_list)

@app.post("/cardio")
def cardiopathy_predictor(s:HeartTestSample):
  json_sample = jsonable_encoder(s)
  sample_list = list(json_sample.values())
  return cardiopathie.heart_disease_predictor()
