import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb

df = pd.read_csv('./input/train_V2.csv', header=0)
test_df = pd.read_csv('./input/test_V2.csv', header=0)

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(df.shape)

# remove outliers
df = df[
    (df.killStreaks < 8) & (df.assists < 10) & (df.headshotKills < 11) & (df.DBNOs < 16) & (df.damageDealt <= 2000) & (
        df.kills < 21)]

# add new features
df['total_distance'] = df['swimDistance'] + df['walkDistance'] + df['rideDistance']
df['kill_and_assist'] = df['kills'] + df['assists']
df['headshot_rate'] = df['headshotKills'] / df['kills']
df['headshot_rate'].fillna(0, inplace=True)

# to use lightgbm, change string match type to int
# to make sure the model can run correctly on test set, do the operation on test set.
matchtypes = pd.concat([df.matchType, test_df.matchType]).unique()
match_dict = {}
for i, each in enumerate(matchtypes):
    match_dict[each] = i
df.matchType = df.matchType.map(match_dict)
test_df.matchType = test_df.matchType.map(match_dict)

# split training set into label and features
label = df['winPlacePerc']
features = df.drop(['Id', 'groupId', 'matchId', 'winPlacePerc'], axis=1)


# split training and valid data


def lightgbm(features, label):
    # transform data to lgb dataset
    train_features, valid_features, train_label, valid_label = train_test_split(features, label)
    train_data = lgb.Dataset(train_features, label=train_label, categorical_feature=['matchType'])
    valid_data = lgb.Dataset(valid_features, label=valid_label, categorical_feature=['matchType'])
    # lgb param
    # see http://lightgbm.apachecn.org/cn/latest/Parameters.html
    parameters = {
        'max_depth': -1, 'min_data_in_leaf': 20, 'feature_fraction': 0.80, 'bagging_fraction': 0.8,
        'boosting_type': 'gbdt', 'learning_rate': 0.1, 'num_leaves': 30, 'subsample': 0.8,
        'application': 'regression', 'num_boost_round': 5000, 'zero_as_missing': False,
        'early_stopping_rounds': 100, 'metric': 'mae', 'seed': 2
    }
    model = lgb.train(parameters, train_set=train_data, valid_sets=valid_data, verbose_eval=500)
    return model, model.best_score


# use random forest model
def random_forest(features, label):
    train_features, valid_features, train_label, valid_label = train_test_split(features, label)
    rf_model = RandomForestRegressor(n_jobs=-1, criterion='mae', n_estimators=20)
    rf_model.fit(train_features, train_label)
    rf_valid_result = rf_model.predict(valid_features)
    rf_error = mean_absolute_error(valid_label, rf_valid_result)
    return rf_model, rf_error


# use k-nn model
def knn(features, label):
    train_features, valid_features, train_label, valid_label = train_test_split(features, label)
    knn_model = KNeighborsRegressor(n_jobs=-1, n_neighbors=10)
    knn_model.fit(train_features, train_label)
    knn_valid_result = knn_model.predict(valid_features)
    knn_error = mean_absolute_error(valid_label, knn_valid_result)
    return knn_model, knn_error


model, score = lightgbm(features, label)
# model, score = random_forest(features, label)
print(score)
# print(test_df.columns)

# predict the value of test data and generate output file
test_id = test_df['Id']

# add features on test set
test_df['total_distance'] = test_df['swimDistance'] + test_df['walkDistance'] + test_df['rideDistance']
test_df['kill_and_assist'] = test_df['kills'] + test_df['assists']
test_df['headshot_rate'] = test_df['headshotKills'] / test_df['kills']
test_df['headshot_rate'].fillna(0, inplace=True)

test_features = test_df.drop(['Id', 'groupId', 'matchId'], axis=1)
test_predict = model.predict(test_features)
# # print(test_predict)
test_predict = test_predict.reshape((len(test_predict), 1))
test_predict = pd.DataFrame(test_predict, columns=['winPlacePerc'])
# print(test_predict)
result = pd.concat([test_id, test_predict], axis=1)
# print(result)
result.to_csv('./submission.csv', index=False)
