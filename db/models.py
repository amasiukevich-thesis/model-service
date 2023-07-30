from sqlalchemy import (
    BigInteger, TIMESTAMP, String, Column, DECIMAL
)
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class Rates(Base):

    __tablename__ = 'rates'

    unix = Column(BigInteger, autoincrement=False, primary_key=True)
    rate_date = Column(TIMESTAMP)
    symbol = Column(String)
    price_open = Column(DECIMAL)
    price_close = Column(DECIMAL)
    price_low = Column(DECIMAL)
    price_high = Column(DECIMAL)
    volume_crypto = Column(DECIMAL)
    volume_usd = Column(DECIMAL)


class PredictionRates(Base):

    __tablename__ = 'prediction_rates'

    unix = Column(BigInteger, autoincrement=False, primary_key=True)
    rate_date = Column(TIMESTAMP)
    symbol = Column(String)
    price_open = Column(DECIMAL)
    price_close = Column(DECIMAL)
    price_low = Column(DECIMAL)
    price_high = Column(DECIMAL)
    volume_crypto = Column(DECIMAL)
    volume_usd = Column(DECIMAL)

