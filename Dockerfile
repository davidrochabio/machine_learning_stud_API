FROM python:3.8
ADD /code /ml_api
ADD requirements.txt /ml_api
WORKDIR /ml_api
RUN pip install -r requirements.txt
WORKDIR /ml_api/app
EXPOSE 8000
CMD python -m uvicorn main:app --host 0.0.0.0