import requests
import json

# dictionary to post data and get the prediction
to_predict = {
    "age": 21,
    "Medu": 1,
    "Fedu": 1,
    "traveltime": 1,
    "studytime": 1,
    "failures": 3,
    "famrel": 5,
    "freetime": 5,
    "goout": 3,
    "Dalc": 3,
    "Walc": 3,
    "health": 3,
    "abscences": 3,
    "G1": 70,
    "G2": 70
}

# dumps the dict into a json to make the post
to_predict = json.dumps(to_predict)

print(type(to_predict))

# get to retrive the model's features
url_features = 'http://127.0.0.1:8000/features/'
feats = requests.get(url_metrics)
print(feats.json())

# post to get predictions
url_prediction = 'http://127.0.0.1:8000/predict/'
preds = requests.post(url_prediction, data=to_predict)
print(preds.json())

# get to retrive the model's metrics
url_metrics = 'http://127.0.0.1:8000/metrics/'
mets = requests.get(url_metrics)
print(mets.json())
