import uvicorn
from services.web_service import app




if __name__ == '__main__':

    # TODO: inject services here
    uvicorn.run(app, host="0.0.0.0", port=8000)
