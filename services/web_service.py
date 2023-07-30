from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from src.helpers import (
    PricePredictionService,
    TooFarFutureException
)

import datetime
import logging


app = FastAPI()

prediction_service = PricePredictionService()


@app.exception_handler(TooFarFutureException)
def too_far_future_exception_handler(request, exc):
    return JSONResponse(content=jsonable_encoder({"error": exc.message}), status_code=403)


@app.get('/version')
def get_version():
    return "<h1>" + str("Hi in the version 1.0 of model service") + "</h1>"


@app.post('/predict')
def predict(input_date: datetime.date):

    logging.info(f"Date: {datetime.datetime.strftime(input_date, '%Y-%m-%d')}")

    input_date = datetime.datetime(input_date.year, input_date.month, input_date.day)

    result = prediction_service.make_prediction(input_date)

    return {'date': datetime.datetime.strftime(input_date, "%Y-%m-%d"), 'prediction': result}






