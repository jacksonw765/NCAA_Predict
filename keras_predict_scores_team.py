import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import *
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import math

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

columns_drop = ['away_losses', 'away_wins', 'conference_losses', 'conference_wins', 'games_played', 'home_losses',
                'home_wins', 'losses', 'wins',  'away_losses_2', 'away_wins_2', 'conference_losses_2', 'conference_wins_2',
                'games_played_2', 'home_losses_2', 'home_wins_2', 'losses_2', 'wins_2',]


def get_game(home_team, away_team, location=2):
    df_teams = pd.read_csv('teams2.csv')
    df_teams_columns = df_teams.columns.tolist()
    df_teams_columns_new = []
    for x in df_teams_columns:
        df_teams_columns_new.append(x + "_2")
    #location = pd.DataFrame([2, 0], columns=['location'])
    team_1 = df_teams[df_teams['abbreviation'] == home_team]
    team_2 = df_teams[df_teams['abbreviation'] == away_team]
    team_2.columns = df_teams_columns_new
    append_team = team_1.reset_index(drop=True).merge(team_2.reset_index(drop=True), left_index=True, right_index=True)
    location_home, location_away, location_neutral = 0, 0, 0
    if location == 1:
        location_neutral = 1
    elif location == 0:
        location_away = 1
    else:
        location_home = 1
    append_team = append_team.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0', 'Unnamed: 0_2'])
    append_team = append_team.drop(columns=columns_drop, errors='ignore')
    location_data = [location_home, location_away, location_neutral]
    location_df = pd.DataFrame([location_data], columns=['location_home', 'location_away', 'location_neutral'])
    append_team = append_team.reset_index(drop=True).merge(location_df.reset_index(drop=True), left_index=True,
                                                           right_index=True)
    return append_team


master_df = pd.read_csv('team_list_scores2021.csv')
y = master_df[['points_for', 'points_against']].to_numpy().tolist()
master_df = master_df.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2', 'points_for', 'points_against'])
#master_df = master_df.filter(regex='percentage')
master_df = master_df.drop(columns=columns_drop)
X = np.round(master_df.to_numpy(), 2).tolist()
preds = []
away_team = "TENNESSEE"
home_team = 'COLORADO'
predict_X = get_game(home_team, away_team)
for x in range(0, 100):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = LinearRegression()
    # model = Pipeline([
    #     ('impute', SimpleImputer(strategy='mean')),
    #     ('scale', MinMaxScaler())
    # ])
    results = model.fit(X_train, y_train)
    #print(model.score(X_train, y_train))

    predictions = model.predict(predict_X.to_numpy())
    pred = np.round(predictions, 0).tolist()
    #test_set_r2 = r2_score(y_test, predictions)
    preds.append(pred[0])

df = pd.DataFrame(preds, columns=[home_team, away_team])
print(home_team + ": " + str(round(df[home_team].mean(), 0)))
print(away_team + ": " + str(round(df[away_team].mean(),0)))