import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('C:/Users/qwert/Downloads/prepped_churn_data.csv')
features = tpot_data.drop(['Churn_Yes', 'Churn_No','customerID'], axis=1)

training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Churn_Yes'], random_state=42)

# Average CV score on the training set was: 0.6243012440455719
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.001, max_depth=2, min_child_weight=9, n_estimators=100, n_jobs=1, subsample=1.0, verbosity=0)),
    StackingEstimator(estimator=RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.7500000000000001, min_samples_leaf=9, min_samples_split=6, n_estimators=100)),
    GaussianNB()
)
exported_pipeline.fit(training_features, training_target)

test_data = pd.read_csv('C:/Users/qwert/Downloads/new_churn_data_unmodified.csv')
test_data = test_data.drop(['customerID'],axis=1)
test_data['LogTotalCharges'] = np.log(test_data['TotalCharges'])
if 'PaymentMethod_Bank transfer (automatic)' not in test_data.columns.values.tolist():
    test_data['PaymentMethod_Bank transfer (automatic)'] = 0
test_data = pd.get_dummies(data=test_data)

results = exported_pipeline.predict(test_data)
print(results)