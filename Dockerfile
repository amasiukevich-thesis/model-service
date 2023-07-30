FROM python:3.10

COPY ./requirements.txt requirements.txt

RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

WORKDIR /app
COPY ./config/* config/
COPY ./db/* db/
COPY ./ml_models/ ml_models/
COPY ./services/* services/
COPY ./src/* src/

COPY main.py main.py

CMD ["python3", "main.py"]
