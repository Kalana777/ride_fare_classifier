from catboost import CatBoostRegressor,CatBoostClassifier
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from  sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

DATA_PATH = Path.cwd().parent /"data"
SEED = 7

def calculate_distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


data_df = pd.read_csv(
    DATA_PATH / "train.csv",
    index_col="tripid"
)

features = [feat for feat in list(data_df) if feat != "label"]

features_df = data_df[features]
labels_df = data_df[["label"]]

labels_df =labels_df.replace(to_replace="correct",value=1)
labels_df =labels_df.replace(to_replace="incorrect",value=0)

features_df["distance"] =  calculate_distance(features_df["pick_lat"],features_df["pick_lon"],features_df["drop_lat"],features_df["drop_lon"])
# features_df["distance"] = (features_df["pick_lat"] - features_df["drop_lat"]) ** 2 + (
#             features_df["pick_lon"] - features_df["drop_lon"]) ** 2

features_df["pickup_time"] = pd.to_datetime(features_df["pickup_time"],errors = "coerce")
features_df["drop_time"] = pd.to_datetime(features_df["drop_time"],errors = "coerce")
features_df['duration'] = features_df['duration'].fillna((features_df['drop_time'] - features_df['pickup_time']).astype('timedelta64[s]'))
features_df['additional_fare'] = features_df['additional_fare'].fillna(features_df['additional_fare'].mode().iloc[0])

features_df["pickup_time_hour"] = features_df["pickup_time"].dt.hour
# features_df["pickup_time_minute"] = features_df["pickup_time"].dt.minute
features_df["drop_time_hour"] =features_df["drop_time"].dt.hour
# features_df["drop_time_minute"] =features_df["drop_time"].dt.minute
# features_df["pickup_time_day"] = features_df["pickup_time"].dt.day
# features_df["drop_time_day"] = features_df["drop_time"].dt.day
# features_df["pickup_time_month"] = features_df["pickup_time"].dt.month
# features_df["drop_time_month"] = features_df["drop_time"].dt.month
# features_df["pick_up_year"] = features_df["drop_time"].dt.year
# features_df["drop_year"] = features_df["drop_time"].dt.year


features_df["effective_time"] = features_df["duration"]-features_df["meter_waiting"]
# features_df.loc[features_df['fare'] <= 2000, 'fare_outlier'] = 1
# features_df.loc[features_df['fare'] > 2000, 'fare_outlier'] = 0
features_df['meter_waiting_fare_diff'] = features_df['meter_waiting_fare'] - features_df['meter_waiting']*0.057
features_df['mean_fare'] = (features_df['fare'] - features_df['meter_waiting_fare'])/(features_df['duration'] - features_df['meter_waiting'])

features_df['pick_day_of_weekk'] = (features_df['pickup_time'] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
features_df['drop_day_of_weekk'] = (features_df['drop_time'] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

tmp = []
for i in features_df['pick_day_of_weekk']:
    dt = datetime.utcfromtimestamp(i)
    tmp.append(dt.weekday())

features_df['pick_day_of_week'] = tmp

tmp = []
for i in features_df['drop_day_of_weekk']:
    dt = datetime.utcfromtimestamp(i)
    tmp.append(dt.weekday())

features_df['drop_day_of_week'] = tmp

features_df = features_df.drop(columns=['pick_day_of_weekk', 'drop_day_of_weekk'])

# features_df['is_weekend'] = np.where(features_df['pick_day_of_week']==6 or 5 or (features_df['pick_day_of_week'] == 4 and features_df['drop_day_of_week'] == 5), 1, 0)
# features_df['is_weekend'] = np.where(features_df['pick_day_of_week'] == 6, 1, np.where(features_df['pick_day_of_week'] == 5, 1, 0))

features_df['pick_day_of_week'] = np.where(features_df['pick_day_of_week'] == 0,'mon', np.where(features_df['pick_day_of_week'] == 1, 'tue', np.where(features_df['pick_day_of_week'] == 2, 'wed', np.where(features_df['pick_day_of_week'] == 3, 'thu', np.where(features_df['pick_day_of_week'] == 4, 'fri', np.where(features_df['pick_day_of_week'] == 5, 'sat', 'sun'))))))
features_df['drop_day_of_week'] = np.where(features_df['drop_day_of_week'] == 0,'mon', np.where(features_df['drop_day_of_week'] == 1, 'tue', np.where(features_df['drop_day_of_week'] == 2, 'wed', np.where(features_df['drop_day_of_week'] == 3, 'thu', np.where(features_df['drop_day_of_week'] == 4, 'fri', np.where(features_df['drop_day_of_week'] == 5, 'sat', 'sun'))))))

features_df["effective_fare"] = features_df["fare"] + features_df["additional_fare"]
features_df["mean_fare_for_unit_length"] = features_df["fare"]/features_df["distance"]
features_df["duration_fare"] = features_df["duration"]*features_df["fare"]

# features_df["unaccounted_fare"] = features_df["additional_fare"] - features_df["meter_waiting_fare"]

# features_df["effective_time"] = (features_df["effective_time"] -features_df["effective_time"].mean() )/features_df["effective_time"].std()
# features_df["effective_fare"] = (features_df["effective_fare"] -features_df["effective_fare"].mean() )/features_df["effective_fare"].std()
# features_df["mean_fare_for_unit_length"] = (features_df["mean_fare_for_unit_length"] -features_df["mean_fare_for_unit_length"].mean() )/features_df["mean_fare_for_unit_length"].std()
# features_df["duration_fare"] = (features_df["duration_fare"] -features_df["duration_fare"].mean() )/features_df["duration_fare"].std()

# features_df = features_df.drop(columns=['drop_day_of_week', 'pick_day_of_week'])#, 'drop_time_day', 'drop_time_hour', 'pickup_time_minute'])
features_df = features_df.drop(
    columns=['drop_day_of_week', 'pick_day_of_week', "drop_lon", 'meter_waiting_fare',
             'pickup_time_hour','drop_lat', 'drop_time_hour','pick_lon'])  # , 'drop_time_day', 'drop_time_hour', 'pickup_time_minute'])

# cat_features = [features_df.columns.get_loc('pick_day_of_week'), features_df.columns.get_loc('drop_day_of_week')]


# features_df['night_job'] = np.where(features_df['pickup_time_hour'] >= 19, 1, np.where(features_df['pickup_time_hour'] <= 5, 1, 0))

print(features_df.dtypes)
# print(cat_features)

X_train, X_eval, y_train, y_eval = train_test_split(
    features_df,
    labels_df,
    test_size=0.33,
    shuffle=True,
    stratify=labels_df,
    random_state=SEED,
)

params1 = {'loss_function':'LogLoss', # objective function
           'iterations': 1600,
          'eval_metric':'F1', # metric
          # 'eval_metric':'TotalF1', # metric
          # 'cat_features': cat_features,
          'verbose': 200, # output to stdout info about training process every 200 iterations
          'random_seed': SEED
         }

# cbc_1 = CatBoostClassifier(**params1)
# # cbc_1.fit(X_train, y_train, # data to train on (required parameters, unless we provide X as a pool object, will be shown below)
# cbc_1.fit(features_df, labels_df, # data to train on (required parameters, unless we provide X as a pool object, will be shown below)
#           # eval_set=(X_eval, y_eval), # data to validate on
#           use_best_model=True, # True if we don't want to save trees created after iteration with the best validation score
#           plot=False # True for visualization of the training process (it is not shown in a published kernel - try executing this code)
#          );
# #
# cbc_1.save_model("./models/demo/last_1_1600")

# feat_import = [t for t in zip(features_df, cbc_1.get_feature_importance())]
# feat_import_df = pd.DataFrame(feat_import, columns = ['feature','VarImp'])
# feat_import_df = feat_import_df.sort_values('VarImp', ascending = False)
# print(feat_import_df)

# model = CatBoostClassifier(iterations=500000,nan_mode='Max')
# model.fit(features_df,labels_df,plot=False)


def load_n_predict():
    from_file = CatBoostClassifier()
    fare_predictor = from_file.load_model("./models/demo/last_1_1600")

    test_features_df = pd.read_csv(DATA_PATH / "test.csv",
                                      index_col="tripid")

    test_features_df["distance"] = calculate_distance(test_features_df["pick_lat"], test_features_df["pick_lon"],
                                            test_features_df["drop_lat"], test_features_df["drop_lon"])
    # test_features_df["distance"] =(test_features_df["pick_lat"]-test_features_df["drop_lat"])**2 + (test_features_df["pick_lon"]-test_features_df["drop_lon"])**2

    test_features_df["pickup_time"] = pd.to_datetime(test_features_df["pickup_time"], errors="coerce")
    test_features_df["drop_time"] = pd.to_datetime(test_features_df["drop_time"], errors="coerce")
    test_features_df["pickup_time_hour"] = test_features_df["pickup_time"].dt.hour
    # test_features_df["pickup_time_minute"] = test_features_df["pickup_time"].dt.minute
    test_features_df["drop_time_hour"] = test_features_df["drop_time"].dt.hour
    # test_features_df["drop_time_minute"] = test_features_df["drop_time"].dt.minute
    # test_features_df["pickup_time_day"] = test_features_df["pickup_time"].dt.day
    # test_features_df["drop_time_day"] = test_features_df["drop_time"].dt.day

    test_features_df["effective_time"] = test_features_df["duration"] - test_features_df["meter_waiting"]
    # test_features_df.loc[test_features_df['fare'] <= 2000, 'fare_outlier'] = 1
    # test_features_df.loc[test_features_df['fare'] > 2000, 'fare_outlier'] = 0
    test_features_df['meter_waiting_fare_diff'] = test_features_df['meter_waiting_fare'] - test_features_df[
        'meter_waiting'] * 0.057
    test_features_df['mean_fare'] = (test_features_df['fare'] - test_features_df['meter_waiting_fare']) / (
            test_features_df['duration'] - test_features_df['meter_waiting'])



    # test_features_df["pickup_time_month"] = test_features_df["pickup_time"].dt.month
    # test_features_df["drop_time_month"] = test_features_df["drop_time"].dt.month
    # test_features_df["pick_up_year"] = test_features_df["drop_time"].dt.year
    # test_features_df["drop_year"] = test_features_df["drop_time"].dt.year

    test_features_df['pick_day_of_weekk'] = (test_features_df['pickup_time'] - np.datetime64(
        '1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    test_features_df['drop_day_of_weekk'] = (test_features_df['drop_time'] - np.datetime64(
        '1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

    tmp = []
    for i in test_features_df['pick_day_of_weekk']:
        dt = datetime.utcfromtimestamp(i)
        tmp.append(dt.weekday())

    test_features_df['pick_day_of_week'] = tmp

    tmp = []
    for i in test_features_df['drop_day_of_weekk']:
        dt = datetime.utcfromtimestamp(i)
        tmp.append(dt.weekday())

    test_features_df['drop_day_of_week'] = tmp

    test_features_df = test_features_df.drop(columns=['pick_day_of_weekk', 'drop_day_of_weekk'])

    # test_features_df["unaccounted_fare"] = test_features_df["additional_fare"] - test_features_df["meter_waiting_fare"]

    # test_features_df['is_weekend'] = np.where(test_features_df['pick_day_of_week'] == 6, 1,
    #                                      np.where(test_features_df['pick_day_of_week'] == 5, 1, 0))

    test_features_df['pick_day_of_week'] = np.where(test_features_df['pick_day_of_week'] == 0, 'mon',
                                               np.where(test_features_df['pick_day_of_week'] == 1, 'tue',
                                                        np.where(test_features_df['pick_day_of_week'] == 2, 'wed',
                                                                 np.where(test_features_df['pick_day_of_week'] == 3, 'thu',
                                                                          np.where(test_features_df['pick_day_of_week'] == 4,
                                                                                   'fri', np.where(
                                                                                  test_features_df['pick_day_of_week'] == 5,
                                                                                  'sat', 'sun'))))))
    test_features_df['drop_day_of_week'] = np.where(test_features_df['drop_day_of_week'] == 0, 'mon',
                                               np.where(test_features_df['drop_day_of_week'] == 1, 'tue',
                                                        np.where(test_features_df['drop_day_of_week'] == 2, 'wed',
                                                                 np.where(test_features_df['drop_day_of_week'] == 3, 'thu',
                                                                          np.where(test_features_df['drop_day_of_week'] == 4,
                                                                                   'fri', np.where(
                                                                                  test_features_df['drop_day_of_week'] == 5,
                                                                                  'sat', 'sun'))))))

    # test_features_df['night_job'] = np.where(test_features_df['pickup_time_hour'] >= 19, 1,
    #                                     np.where(test_features_df['pickup_time_hour'] <= 5, 1, 0))

    test_features_df["effective_fare"] = test_features_df["fare"] + test_features_df["additional_fare"]
    test_features_df["mean_fare_for_unit_length"] = test_features_df["fare"]/test_features_df["distance"]
    test_features_df["duration_fare"] = test_features_df["duration"]*test_features_df["fare"]

    # test_features_df["effective_time"] = (test_features_df["effective_time"] -test_features_df["effective_time"].mean() )/test_features_df["effective_time"].std()
    # test_features_df["effective_fare"] = (test_features_df["effective_fare"] -test_features_df["effective_fare"].mean() )/test_features_df["effective_fare"].std()
    # test_features_df["mean_fare_for_unit_length"] = (test_features_df["mean_fare_for_unit_length"] -test_features_df["mean_fare_for_unit_length"].mean() )/test_features_df["mean_fare_for_unit_length"].std()
    # test_features_df["duration_fare"] = (test_features_df["duration_fare"] -test_features_df["duration_fare"].mean() )/test_features_df["duration_fare"].std()

    test_features_df = test_features_df.drop(columns=['drop_day_of_week', 'pick_day_of_week', "drop_lon", 'meter_waiting_fare', 'pickup_time_hour','drop_lat', 'drop_time_hour','pick_lon']) #, 'drop_time_day', 'drop_time_hour', 'pickup_time_minute'])

    fare_preds = fare_predictor.predict(test_features_df)

    submission_df = pd.read_csv(DATA_PATH / "sample_submission.csv",
                                index_col="tripid")
    print(submission_df.head())
    submission_df["prediction"] = fare_preds
    print(submission_df.head())

    submission_df.to_csv('./results/last_1_1600.csv', index=True)

    feat_import = [t for t in zip(features_df, fare_predictor.get_feature_importance())]
    feat_import_df = pd.DataFrame(feat_import, columns=['feature', 'VarImp'])
    feat_import_df = feat_import_df.sort_values('VarImp', ascending=False)
    print(feat_import_df)

load_n_predict()