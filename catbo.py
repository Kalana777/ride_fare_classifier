from catboost import CatBoostRegressor,CatBoostClassifier
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from  sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

DATA_PATH = Path.cwd().parent /"data"
SEED = 7

data_df = pd.read_csv(
    DATA_PATH / "train.csv",
    index_col="tripid"
)

features = [feat for feat in list(data_df) if feat != "label"]

features_df = data_df[features]
labels_df = data_df[["label"]]


numeric_many_preprocessing_steps = Pipeline([
    ('simple_imputer', SimpleImputer(strategy='mean')),
    # ('standard_scaler', StandardScaler())
])

numeric_one_preprocessing_steps = Pipeline([
    ('simple_imputer', SimpleImputer(strategy='most_frequent')),
    # ('standard_scaler', StandardScaler())
])

# categorical_preprocessing_steps = Pipeline([
#     # ('simple_imputer', SimpleImputer(strategy='most_frequent')),
#     ('fillna', fillna),
#     # ('onehot', OneHotEncoder(handle_unknown='ignore')),
# ])


features_df['pickup_time'].fillna(method ='ffill')
features_df['drop_time'].fillna(method ='ffill')


month, day, year, hour, minu = [], [], [], [], []

for date in features_df['pickup_time']:
    temp1 = date.split('/')
    month.append(float(temp1[0]))
    day.append(float(temp1[1]))
    temp2 = temp1[2].split()
    year.append(float(temp2[0]))
    temp3 = temp2[1].split(':')
    hour.append(float(temp3[0]))
    minu.append(float(temp3[1]))

features_df['pickup_month'] = month
features_df['pickup_day'] = day
features_df['pickup_year'] = year
features_df['pickup_hour'] = hour
features_df['pickup_min'] = minu

month, day, year, hour, minu = [], [], [], [], []

for date in features_df['drop_time']:
    temp1 = date.split('/')
    month.append(float(temp1[0]))
    day.append(float(temp1[1]))
    temp2 = temp1[2].split()
    year.append(float(temp2[0]))
    temp3 = temp2[1].split(':')
    hour.append(float(temp3[0]))
    minu.append(float(temp3[1]))

features_df['drop_month'] = month
features_df['drop_day'] = day
features_df['drop_year'] = year
features_df['drop_hour'] = hour
features_df['drop_min'] = minu

# features_df = features_df.drop(columns=['pickup_time', 'drop_time'])

numeric_cols_many = list(features_df.columns)
print(features_df.columns)
numeric_cols_many.remove('additional_fare')
numeric_cols_many.remove( 'drop_time')
numeric_cols_many.remove('pickup_time')
numeric_cols_one = ['additional_fare']


preproc = ColumnTransformer(
    transformers = [
        ("numeric_many", numeric_many_preprocessing_steps, numeric_cols_many),
        ("numeric_one", numeric_one_preprocessing_steps, numeric_cols_one),
        # ("categorical", categorical_preprocessing_steps, categorical_cols),
        # ("numeric", numeric_preprocessing_steps, numeric_cols),

    ],
    remainder = "passthrough"
)

features_dff = preproc.fit_transform(features_df)

cat_features = [20,21]
labs = []
for lab in labels_df['label']:
    if lab == "correct":
        labs.append(1)
    else:
        labs.append(0)

labels_dff = pd.DataFrame(
    {
        # "label": np.where(labels_df['label'] == 'çorrect', 1,0)
        "label": labs
    }
)

X_train, X_eval, y_train, y_eval = train_test_split(
    features_dff,
    labels_dff,
    test_size=0.33,
    shuffle=True,
    stratify=labels_dff,
    random_state=SEED,
)

params1 = {'loss_function':'CrossEntropy', # objective function
           'iterations': 10000,
          'eval_metric':'F1', # metric
          # 'eval_metric':'TotalF1', # metric
          'cat_features': cat_features,
          'verbose': 200, # output to stdout info about training process every 200 iterations
          'random_seed': SEED
         }

print(labels_dff)

#
cbc_1 = CatBoostClassifier(**params1)
cbc_1.fit(X_train, y_train, # data to train on (required parameters, unless we provide X as a pool object, will be shown below)
# cbc_1.fit(features_dff, labels_dff, # data to train on (required parameters, unless we provide X as a pool object, will be shown below)
          eval_set=(X_eval, y_eval), # data to validate on
          use_best_model=True, # True if we don't want to save trees created after iteration with the best validation score
          plot=False # True for visualization of the training process (it is not shown in a published kernel - try executing this code)
         );
#
cbc_1.save_model("./models/time_splits_with_date_time_too_all_together_21_features")
# #
# # #




def load_n_predict():
    from_file = CatBoostClassifier()
    fare_predictor = from_file.load_model("./models/time_splits_with_date_time_too_all_together_21_features")

    test_features_df = pd.read_csv(DATA_PATH / "test.csv",
                                      index_col="tripid")

    # test_features_df = test_features_df[[f for f in list(features_df) if f not in [ 'drop_lat', 'drop_lon']]]#,'pick_lat', 'pick_lon']]]

    test_features_df['pickup_time'].fillna(method ='ffill')
    test_features_df['drop_time'].fillna(method ='ffill')

    month, day, year, hour, minu = [], [], [], [], []

    for date in test_features_df['pickup_time']:
        temp1 = date.split('/')
        month.append(float(temp1[0]))
        day.append(float(temp1[1]))
        temp2 = temp1[2].split()
        year.append(float(temp2[0]))
        temp3 = temp2[1].split(':')
        hour.append(float(temp3[0]))
        minu.append(float(temp3[1]))

    test_features_df['pickup_month'] = month
    test_features_df['pickup_day'] = day
    test_features_df['pickup_year'] = year
    test_features_df['pickup_hour'] = hour
    test_features_df['pickup_min'] = minu

    month, day, year, hour, minu = [], [], [], [], []

    for date in test_features_df['drop_time']:
        temp1 = date.split('/')
        month.append(float(temp1[0]))
        day.append(float(temp1[1]))
        temp2 = temp1[2].split()
        year.append(float(temp2[0]))
        temp3 = temp2[1].split(':')
        hour.append(float(temp3[0]))
        minu.append(float(temp3[1]))

    test_features_df['drop_month'] = month
    test_features_df['drop_day'] = day
    test_features_df['drop_year'] = year
    test_features_df['drop_hour'] = hour
    test_features_df['drop_min'] = minu


    feat_dff = preproc.fit_transform(test_features_df)

    fare_preds = fare_predictor.predict(feat_dff)

    submission_df = pd.read_csv(DATA_PATH / "sample_submission.csv",
                                index_col="tripid")
    print(submission_df.head())
    submission_df["prediction"] = fare_preds
    print(submission_df.head())

    submission_df.to_csv('time_splits_with_date_time_too_all_together_21_features_5000_crossentropy.csv', index=True)

# load_n_predict()