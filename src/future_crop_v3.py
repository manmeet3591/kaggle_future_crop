import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from autogluon.tabular import TabularPredictor

def load_data(crop: str, mode: str="train"):
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

def get_fourier_components(data, num_components=3):
    fourier = np.fft.fft(data)
    indices = np.arange(len(fourier))
    mask = indices <= num_components
    real_components = np.real(fourier[mask])
    imag_components = np.imag(fourier[mask])
    return np.concatenate([real_components, imag_components])

def get_simple_features(data_dict):
    for variable in ['tas', 'pr', 'rsds', 'tasmax', 'tasmin']:
        data_dict[variable][f'mean_{variable}'] = data_dict[variable].iloc[:, 35:215].mean(axis=1)
        data_dict[variable][f'min_{variable}'] = data_dict[variable].iloc[:, 35:215].min(axis=1)
        data_dict[variable][f'max_{variable}'] = data_dict[variable].iloc[:, 35:215].max(axis=1)
        
        fourier_components = data_dict[variable].iloc[:, 35:215].apply(get_fourier_components, axis=1, result_type='expand')
        for i in range(fourier_components.shape[1] // 2):
            data_dict[variable][f'fourier_{variable}_real_{i+1}'] = fourier_components.iloc[:, i]
            data_dict[variable][f'fourier_{variable}_imag_{i+1}'] = fourier_components.iloc[:, i + (fourier_components.shape[1] // 2)]

    data_dict['features'] = data_dict['soil_co2'][['year', 'crop', 'texture_class', 'co2', 'nitrogen']]

    for variable in ['tas', 'pr', 'rsds', 'tasmax', 'tasmin']:
        data_dict['features'] = data_dict['features'].join(data_dict[variable][[f'mean_{variable}', f'min_{variable}', f'max_{variable}']])
        for i in range(fourier_components.shape[1] // 2):
            data_dict['features'] = data_dict['features'].join(data_dict[variable][[f'fourier_{variable}_real_{i+1}', f'fourier_{variable}_imag_{i+1}']])

    data_dict['features']['pr_sum'] = data_dict['pr'].iloc[:, 35:215].sum(axis=1)
    return data_dict['features']

# Loading and processing data
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

y_train = pd.concat([maize_train['target'][maize_train['features']['year']<410],
                     wheat_train['target'][wheat_train['features']['year']<410]])['yield']
y_test = pd.concat([maize_train['target'][maize_train['features']['year']>=410],
                    wheat_train['target'][wheat_train['features']['year']>=410]])['yield']

# Combine x_train and y_train for AutoGluon
train_data = x_train.copy()
train_data['yield'] = y_train

# Train AutoGluon model
predictor = TabularPredictor(label='yield').fit(train_data)

# Evaluate the model
y_pred = predictor.predict(x_test)
print('MSE: ', mean_squared_error(y_test, y_pred, squared=False))

# Train on full training data
train_data_full = pd.concat([x_train, x_test])
train_data_full['yield'] = pd.concat([y_train, y_test])
predictor_full = TabularPredictor(label='yield').fit(train_data_full)

# Load and process test data
maize_test = load_data("maize", "test")
wheat_test = load_data("wheat", "test")

maize_test['features'] = get_simple_features(maize_test)
wheat_test['features'] = get_simple_features(wheat_test)

x_final = pd.concat([maize_test['features'].drop(columns=['year']),
                     wheat_test['features'].drop(columns=['year'])])

x_final['crop'] = x_final['crop'].map({'maize' : 0, 'wheat' : 1})

# Make final predictions
final_pred = predictor_full.predict(x_final)
x_final['yield'] = final_pred

# Save predictions to a CSV file
x_final[['yield']].to_csv('submission.csv')
