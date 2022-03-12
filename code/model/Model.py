import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import max_error
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy
import pickle
import pathlib
import os


# NOTE: Make sure that the outcome column is labeled 'target' in the data file
data = pd.read_csv(pathlib.Path(__file__).parent.absolute().__str__() + '/student-mat.csv', sep=';')
data['target'] = data['G3']
data.drop(columns='G3', inplace=True)
features = data.drop('target', axis=1).select_dtypes([np.number])
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, data['target'], random_state=25)
features = features.dtypes.to_dict()

# Instantiate model
model = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        StackingEstimator(estimator=RidgeCV())
    ),
    XGBRegressor(learning_rate=0.1, max_depth=2, min_child_weight=9, n_estimators=1000, nthread=1,
                 objective="reg:squarederror", subsample=0.35000000000000003)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(model.steps, 'random_state', 25)
Model = model.fit(training_features, training_target)

# Building Metrics
testing_pred = Model.predict(testing_features)
score = model.score(testing_features, testing_target)
mse = MSE(testing_target, testing_pred)
rmse = mse**(1/2)
max_error = max_error(testing_target, testing_pred)
eval_metrics_dict = {'r2':score, 'mse': mse, 'rmse': rmse, 'max_error': max_error}
# results = model.predict(testing_features)
# model.predict(
#     data[testing_features.columns]) - data['target']

# data['predicted'] = model.predict(data[testing_features.columns])
# data['residuals'] = data['target'] - data['predicted']

# creates a dictionary to implement the pickle files
pickle_names = {'features': features, 
                'model': Model,
                'metrics': eval_metrics_dict}

# saves the pickle files
for i,j in pickle_names.items():
    pickle_path = os.path.dirname(os.path.abspath(__file__)).replace('model', 'app\\data\\' + i + '.pickle')

    pickle.dump(j, open(pickle_path, 'wb'))
