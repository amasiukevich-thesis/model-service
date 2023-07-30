import os
import onnxruntime as ort

import numpy as np

from datetime import datetime, timedelta

from sqlalchemy import create_engine, and_, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert

from typing import List, Dict, Any

# project imports
from db.models import Rates, PredictionRates
import config

import pickle as pkl

TOO_FAR_FUTURE_DAYS = 7


class TooFarFutureException(Exception):

    def __init__(self):
        self.message = "Prediction not supported - too far future"


class PricePredictionService:

    def __init__(self):
        self.inference_session, self.scaler = load_model()
        self.db_session = load_db_session()

    def make_prediction(self, input_date: datetime) -> float:

        """
        A wrapper around the prediction making

        :param input_date: A date for which the prediction is needed
        :return: float value of the prediction - either taken from the DB (the past) or predicted by the model (future)
        :raises TooFarFutureException: If the prediction is > 7 days after the maximum value in the database
        """

        rates_max_date = self.db_session.query(func.max(Rates.rate_date)).all()[0][0]
        if rates_max_date > input_date:

            # returning the real value from the past
            needed_row = self.db_session.query(Rates).filter(Rates.rate_date == input_date).all()

            return round(float(needed_row[0].price_open), 2)

        # TODO: Workaround for not predicting too much into the future
        elif input_date > rates_max_date + timedelta(days=TOO_FAR_FUTURE_DAYS):
            raise TooFarFutureException()

        else:
            input_sequence = self.get_rates_sequence(input_date)
            return self.predict(np.array(input_sequence))

    def get_rates_sequence(self, sent_date: datetime) -> List[float]:

        """
        Returns a sequence of rates which are needed to predict the value for a given date.

        :param sent_date: a date for which to predict
        :return: a sequence of rates which are needed to predict the value for
        """

        # first getting the max date from rates and prediction rates
        rates_max_date = self.db_session.query(func.max(Rates.rate_date)).all()[0][0]
        prediction_rates_max_date = self.db_session.query(func.max(PredictionRates.rate_date)).all()[0][0]

        n_supporting = int((sent_date - rates_max_date) / timedelta(hours=1))

        begin_supporting_date = rates_max_date - timedelta(hours=config.N_FEATURES)

        # actual rates got from the database
        rates = self.db_session.query(Rates).filter(Rates.rate_date > begin_supporting_date).order_by(Rates.rate_date).all()

        sequence = [{"open": rate.price_open, "symbol": rate.symbol, "date": rate.rate_date} for rate in rates]

        # TODO: A workaround for now - only predicts open price
        predicted = []

        for i in range(n_supporting):

            symbol = sequence[0]['symbol']
            features = [item['open'] for item in sequence[-config.N_FEATURES:]]
            latest_date = sequence[-1]['date']
            new_rate = self.predict(np.array(features))
            predicted_value = {'open': new_rate, 'date': latest_date + timedelta(hours=1), 'symbol': symbol}

            predicted.append(predicted_value)
            sequence.append(predicted_value)

        # TODO Update this not for every prediction
        self.fill_autoreg_predictions(predicted)

        return [item['open'] for item in sequence[-config.N_FEATURES:]]

    def fill_autoreg_predictions(self, predicted: List[Dict[str, object]]) -> None:

        """
        Inserts the generated predictions to the database

        :param predicted: Generated predictions as List of Dictionaries
        :return: None
        """

        for row in predicted:

            row_to_insert: dict[str, object | float | Any] = {
                'unix': int(datetime.timestamp(row['date']) * 10 ** 9),
                'price_open': float(row['open']),
                'price_close': float(row['open']),
                'price_high': float(row['open']),
                'price_low': float(row['open']),
                'rate_date': row['date'],
                'symbol': row['symbol'],
                'volume_usd': 0,
                'volume_crypto': 0
            }
            row_object = PredictionRates(**row_to_insert)

            # UPSERTS values in the database
            self.db_session.merge(row_object)
            self.db_session.commit()

    def predict(self, rates_sequence: np.ndarray) -> float:

        """
        Predicts the value for the given sequence of rates
        :param rates_sequence:
        :return:
        """

        if len(rates_sequence) != config.N_FEATURES:
            raise Exception("Invalid features length")

        else:
            features = np.array(rates_sequence, dtype=np.float32).reshape(-1, 1)

            scaled_features = self.scaler.transform(features)

            # TODO: Later make this dynamic - number of input features - only 'open' for now
            features = scaled_features.reshape(1, config.N_FEATURES, 1)
            ort_value = ort.OrtValue.ortvalue_from_numpy(features)

            inputs_name = self.inference_session.get_inputs()[0].name
            outputs_name = self.inference_session.get_outputs()[0].name

            results = self.inference_session.run([outputs_name], {inputs_name: ort_value})

            results = self.scaler.inverse_transform(results[0])

        # TODO: beautify this
        return round(float(results[0][0]), 2)

def load_model():

    model_path = os.path.join(config.MODEL_DIR, config.MODEL_VERSION, f"{config.MODEL_CLASS}_model.onnx")
    scaler_path = os.path.join(config.MODEL_DIR, config.MODEL_VERSION, f"scaler.pkl")
    inference_session = ort.InferenceSession(model_path)

    with open(scaler_path, "rb") as fpkl:
        scaler = pkl.load(fpkl)

    return inference_session, scaler


def load_db_session():

    # TODO: separate session making and engine making
    connection_string = os.environ.get("CONNECTION_STRING")
    engine = create_engine(connection_string, pool_size=10, max_overflow=0)

    SessionClass = sessionmaker(bind=engine)
    session = SessionClass()

    return session


if __name__ == '__main__':

    pp_service = PricePredictionService()

    date_today =  datetime.combine(datetime.today(), datetime.min.time())
    date_tomorrow = date_today + timedelta(days=1)

    result_super_dummy = pp_service.predict([1] * 149 + [0])

    input_sequence = pp_service.get_rates_sequence(date_tomorrow)

    result = pp_service.predict(np.array(input_sequence))
    print(result)
