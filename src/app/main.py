import fastapi
from fastapi.middleware.cors import CORSMiddleware

import mlflow
from pydantic import BaseModel, conint
import pandas as pd
import json
import uvicorn

# Load the application configuration
with open('./config/app.json') as f:
    config = json.load(f)


# Define the inputs expected in the request body as JSON
class Request(BaseModel):
    """
    Request model for the API, defining the input structure.

    Attributes:
        LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
        SEX: Gender (1=male, 2=female)
        EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
        MARRIAGE: Marital status (1=married, 2=single, 3=others)
        AGE: Age in years
        PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
        PAY_2: Repayment status in August, 2005 (scale same as above)
        PAY_3: Repayment status in July, 2005 (scale same as above)
        PAY_4: Repayment status in June, 2005 (scale same as above)
        PAY_5: Repayment status in May, 2005 (scale same as above)
        PAY_6: Repayment status in April, 2005 (scale same as above)
        BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
        BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
        BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
        BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
        BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
        BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
        PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
        PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
        PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
        PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
        PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
        PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
    """
    LIMIT_BAL: float = 0
    SEX: int = 1
    EDUCATION: int = 0
    MARRIAGE: int = 0
    AGE: int = 35  
    PAY_0: int = 0
    PAY_2: int = 0
    PAY_3: int = 0
    PAY_4: int = 0
    PAY_5: int = 0
    PAY_6: int = 0
    BILL_AMT1: float = 0
    BILL_AMT2: float = 0
    BILL_AMT3: float = 0
    BILL_AMT4: float = 0
    BILL_AMT5: float = 0
    BILL_AMT6: float = 0
    PAY_AMT1: float = 0
    PAY_AMT2: float = 0
    PAY_AMT3: float = 0
    PAY_AMT4: float = 0
    PAY_AMT5: float = 0
    PAY_AMT6: float = 0

# Create a FastAPI application
app = fastapi.FastAPI()

# Add CORS middleware to allow all origins, methods, and headers for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Set up actions to perform when the app starts.

    Configures the tracking URI for MLflow to locate the model metadata
    in the local mlruns directory.
    """
        
    mlflow.set_tracking_uri(f"{config['tracking_base_url']}:{config['tracking_port']}")

    # Load the registered model specified in the configuration
    model_uri = f"models:/{config['model_name']}@{config['model_version']}"
    app.model = mlflow.pyfunc.load_model(model_uri = model_uri)
    
    print(f"Loaded model {model_uri}")


@app.post("/bom_pagador")
async def predict(input: Request):  
    """
    Prediction endpoint that processes input data and returns a model prediction.

    Parameters:
        input (Request): Request body containing input values for the model.

    Returns:
        dict: A dictionary with the model prediction under the key "prediction".
    """

    # Build a DataFrame from the request data
    input_df = pd.DataFrame.from_dict({k: [v] for k, v in input.model_dump().items()})

    # Predict using the model and retrieve the first item in the prediction list
    prediction = app.model.predict(input_df)

    # Return the prediction result as a JSON response
    return {"prediction": prediction.tolist()[0]}

# Run the app on port 5003
uvicorn.run(app=app, port=config["service_port"], host="0.0.0.0")
