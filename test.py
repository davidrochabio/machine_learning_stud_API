import requests
import json

# dictionary for post the data and get the prediction
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

# post to get predictions
url = 'http://127.0.0.1:8000/predict/'
r = requests.post(url, data=to_predict)
print(r.json())

# get to retrive the model's metrics
url_metrics = 'http://127.0.0.1:8000/metrics/'
m = requests.get(url_metrics)
print(m.json())
