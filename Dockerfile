FROM python:3.8
ADD . /code
WORKDIR /code/app
RUN pip install -r ../requirements.txt
EXPOSE 8000
CMD python -m uvicorn main:app --host 0.0.0.0