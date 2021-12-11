import statistics
import sys

import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.applications.densenet import layers
from keras.optimizer_v2.adam import Adam
from keras.saving.model_config import model_from_json
from sklearn.model_selection import train_test_split
from sportsipy.ncaab.teams import Teams
from sportsipy.ncaaf.teams import Teams as Teamsf
from sklearn.datasets import load_iris
from sportsipy.ncaab.boxscore import Boxscore
from sportsipy.ncaaf.schedule import Schedule
from sportsipy.ncaaf.boxscore import Boxscore as Boxscoref
from sklearn.model_selection import train_test_split

columns_drop = ['away_losses', 'away_wins', 'conference_losses', 'conference_wins', 'games_played', 'home_losses',
                'home_wins', 'losses', 'wins',  'away_losses_2', 'away_wins_2', 'conference_losses_2', 'conference_wins_2',
                'games_played_2', 'home_losses_2', 'home_wins_2', 'losses_2', 'wins_2',]

def get_game(home_team, away_team, location=2):
    df_teams = pd.read_csv('teams_weights2022.csv')
    df_teams_columns = df_teams.columns.tolist()
    df_teams_columns_new = []
    for x in df_teams_columns:
        df_teams_columns_new.append(x + "_2")
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


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


master_df = pd.read_csv('team_list_scores2022.csv')
y = master_df['points_for'].to_numpy().tolist()
master_df = master_df.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2', 'points_for', 'points_against'])
master_df = master_df.drop(columns=columns_drop)
#X = np.round(master_df.to_numpy(), 2).tolist()
X = master_df.to_numpy().tolist()

num_sim = 1
# team1 = "VERMONT"
# team2 = "BROWN"
#
# predict_X_1 = get_game(team1, team2, location=0)
# predict_X_2 = get_game(team2, team1, location=2)
#
def simulate_game(team, predict_data):
    preds = []
    file = open('new_bad_model', 'r')
    model_json = file.read()
    file.close()
    model = model_from_json(model_json)
    model.load_weights('new_bad_model.h5')
    predict_X = predict_data.to_numpy()
    predictions = model.predict([predict_X])
    preds.append(predictions)
    df = pd.DataFrame([preds], columns=[team])
    return int(df[team].mean())
#
# output = team1 + ": " + str(simulate_game(team1, predict_X_1))
# output2 = team2 + ": " + str(simulate_game(team2, predict_X_2))
# print(output)
# print(output2)

teams_1 = [{"APPALACHIAN-STATE": 65}, {'VERMONT': 70}, {"MURRAY-STATE": 74}, {"LOYOLA-IL": 69}, {'DEPAUL': 62}, {"MILWAUKEE": 54}]
teams_2 = [{"FURMAN": 73}, {'BROWN': 65}, {"MEMPHIS": 72}, {'MEMPHIS': 72}, {'LOUISVILLE': 55}, {"COLORADO": 65}]

results = []
score_results = []
for team1_obj, team2_obj in zip(teams_1, teams_2):
    team1 = list(team1_obj.keys())[0]
    team2 = list(team2_obj.keys())[0]
    did_away_win = False
    did_away_win_pred = False
    team1_score = list(team1_obj.values())[0]
    team2_score = list(team2_obj.values())[0]
    if team1_score > team2_score:
        did_away_win = True
    try:
        predict_X_1 = get_game(team1, team2, location=0)
        predict_X_2 = get_game(team2, team1, location=2)
        team1_pred = simulate_game(team1, predict_X_1)
        team2_pred = simulate_game(team2, predict_X_2)
        team1_dif = team1_score - team1_pred
        team2_dif = team2_score - team2_pred
        if team1_pred > team2_pred:
            did_away_win = True
        print(team1 + ": " + str(team1_dif))
        print(team2 + ": " + str(team2_dif))
        result = (did_away_win == did_away_win_pred)
        print(result)
        results.append(result)
        score_results.append(abs(team1_dif))
        score_results.append(abs(team2_dif))
        print('\n')
    except Exception as e:
        print("Failed " + team1, team2)
print(float(results.count(True)/len(results)))
print(str(statistics.mean((score_results))))


