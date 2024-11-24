import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb

import pickle

# Parameters

eta = 0.1
max_depth = 3
min_child_weight = 10

# Data preparation

df = pd.read_csv("Financial-Distress.csv")

df.columns = df.columns.str.lower().str.replace(' ', '_')

df["x80"] = df["x80"].astype(dtype="str")

df = df.drop(columns={"company", "time"})
df["financial_distress"] = (df["financial_distress"]
                            <= np.float64(-0.5)).astype(int)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=39)
df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.financial_distress.values
y_test = df_test.financial_distress.values

del df_full_train['financial_distress']
del df_test['financial_distress']

# Train XGBoost

dicts_full_train = df_full_train.to_dict(orient='records')
dicts_test = df_test.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
X_test = dv.transform(dicts_test)
feature_names = list(dv.get_feature_names_out())
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train,
                         feature_names=feature_names)
dtest = xgb.DMatrix(X_test, feature_names=feature_names)

xgb_params = {
    'eta': eta,
    'max_depth': max_depth,
    'min_child_weight': min_child_weight,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 39,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dfulltrain, num_boost_round=34)
# Evaluate the final model

y_pred = model.predict(dtest)
auc = roc_auc_score(y_test, y_pred)
print(f'AUC on test : {auc}')
# Saving the model to pickle

with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print("Model is saved to model.bin")
