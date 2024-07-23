# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # plotting
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/workspace/future_crop'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def load_data(crop: str, mode: str="train"):
    # note that years represent an offset from model spinup;
    # soil co2 dataset has real year;
    # 0-30 are days before sowing, 31-238 are days after sowing
    tasmax = pd.read_parquet(f"/workspace/future_crop/tasmax_{crop}_{mode}.parquet")
    tasmin = pd.read_parquet(f"/workspace/future_crop/tasmin_{crop}_{mode}.parquet")
    tas = pd.read_parquet(f"/workspace/future_crop/tas_{crop}_{mode}.parquet")
    pr = pd.read_parquet(f"/workspace/future_crop/pr_{crop}_{mode}.parquet")
    rsds = pd.read_parquet(f"/workspace/future_crop/rsds_{crop}_{mode}.parquet")
    soil_co2 = pd.read_parquet(f"/workspace/future_crop/soil_co2_{crop}_{mode}.parquet")
    target = pd.read_parquet(f"/workspace/future_crop/{mode}_solutions_{crop}.parquet") if mode == "train" else None
    return {
        'tasmax': tasmax,
        'tasmin': tasmin,
        'tas': tas,
        'pr': pr,
        'rsds': rsds,
        'soil_co2': soil_co2,
        'target': target,
    }

def get_simple_features(data_dict):
    for variable in ['tas','pr','rsds', 'tasmax', 'tasmin']:
        data_dict[variable][f'mean_{variable}'] = data_dict[variable].iloc[:,35:215].mean(axis=1)
        data_dict[variable][f'min_{variable}'] = data_dict[variable].iloc[:,35:215].min(axis=1)
        data_dict[variable][f'max_{variable}'] = data_dict[variable].iloc[:,35:215].max(axis=1)

    data_dict['features'] = data_dict['soil_co2'][['year','crop','texture_class', 'co2', 'nitrogen']]

    for variable in ['tas','pr','rsds', 'tasmax', 'tasmin']:
        data_dict['features'] = data_dict['features'].join(data_dict[variable][[f'mean_{variable}', f'min_{variable}', f'max_{variable}']])

    data_dict['features']['pr_sum'] = data_dict['pr'].iloc[:,35:215].sum(axis=1)
    return data_dict['features']

maize_train = load_data("maize", "train")

maize_train['features'] = get_simple_features(maize_train)

wheat_train = load_data("wheat", "train")

wheat_train['features'] = get_simple_features(wheat_train)

x_train = pd.concat([maize_train['features'][maize_train['features']['year']<410].drop(columns=['year']),
                     wheat_train['features'][wheat_train['features']['year']<410].drop(columns=['year'])])
x_test = pd.concat([maize_train['features'][maize_train['features']['year']>=410].drop(columns=['year']),
                    wheat_train['features'][wheat_train['features']['year']>=410].drop(columns=['year'])])

x_train['crop'] = x_train['crop'].map({'maize' : 0, 'wheat' : 1})
x_test['crop'] = x_test['crop'].map({'maize' : 0, 'wheat' : 1})

y_train = x_train.join(pd.concat([maize_train['target'], wheat_train['target']]))['yield']
y_test = x_test.join(pd.concat([maize_train['target'], wheat_train['target']]))['yield']

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

model = ExtraTreesRegressor(random_state=0, n_jobs=-1, max_depth=30, n_estimators=30)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('MSE: ', mean_squared_error(y_test, y_pred, squared=False))

model.fit(pd.concat([x_train, x_test]), pd.concat([y_train, y_test]))

maize_test = load_data("maize", "test")
wheat_test = load_data("wheat", "test")

maize_test['features'] = get_simple_features(maize_test)
wheat_test['features'] = get_simple_features(wheat_test)

x_final = pd.concat([maize_test['features'].drop(columns=['year']),
                     wheat_test['features'].drop(columns=['year'])])

x_final['crop'] = x_final['crop'].map({'maize' : 0, 'wheat' : 1})

final_pred = model.predict(x_final)

x_final['yield'] = final_pred

x_final[['yield']].to_csv('submission.csv')