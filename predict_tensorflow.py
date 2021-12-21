import statistics
import sys

import pandas as pd
import numpy as np
from keras.saving.model_config import model_from_json

from get_game_schedule import get_scores_for_date

columns_drop = ['away_losses', 'away_wins', 'conference_losses', 'conference_wins', 'games_played', 'home_losses',
                'home_wins', 'losses', 'wins',  'away_losses_2', 'away_wins_2', 'conference_losses_2', 'conference_wins_2',
                'games_played_2', 'home_losses_2', 'home_wins_2', 'losses_2', 'wins_2' 'team2_minutes_played', 'team1_minutes_played',
                'team1_losses', 'team2_losses', 'team1_win_percentage', 'team2_win_percentage', 'team2_minutes_played']


def get_game(home_team, away_team, location=2):
    df_teams = pd.read_csv('teams_list_boxscores2022.csv').dropna()
    team_1 = df_teams[df_teams['team1'] == home_team].filter(regex='team1').drop(columns=['team1'])\
        .drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2',
                                    'points_for', 'points_against', 'team1', 'team2'], errors='ignore')\
        .drop(columns=columns_drop, errors='ignore')
    team_2 = df_teams[df_teams['team2'] == away_team].filter(regex='team1').drop(columns=['team1'])\
        .drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2',
                                    'points_for', 'points_against', 'team1', 'team2'], errors='ignore')\
        .drop(columns=columns_drop, errors='ignore')
    team_1 = team_1.mean().to_frame().transpose()
    team_2 = team_2.mean().to_frame().transpose()
    append_team = team_1.reset_index(drop=True).merge(team_2.reset_index(drop=True), left_index=True, right_index=True)
    location_home, location_away, location_neutral = 0, 0, 0
    if location == 1:
        location_neutral = 1
    elif location == 0:
        location_away = 1
    else:
        location_home = 1
    location_data = [location_home, location_away, location_neutral]
    location_df = pd.DataFrame([location_data], columns=['location_home', 'location_away', 'location_neutral'])
    append_team = append_team.reset_index(drop=True).merge(location_df.reset_index(drop=True),
                                                           left_index=True, right_index=True)
    append_team = append_team.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2',
                                    'points_for', 'points_against', 'team1', 'team2'], errors='ignore')\
        .drop(columns=columns_drop, errors='ignore')
    return append_team


master_df = pd.read_csv('teams_list_boxscores2022.csv').dropna()
y = master_df['points_for'].to_numpy()
#y = y.reshape(1, y.shape[0],)
master_df = master_df.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2',
                                    'points_for', 'points_against', 'team1', 'team2'], errors='ignore')
master_df = master_df.drop(columns=columns_drop, errors='ignore')
file = open('lstm.json', 'r')
model_json = file.read()
file.close()
model = model_from_json(model_json)
model.load_weights('lstm.h5')
#X = np.round(master_df.to_numpy(), 2).tolist()
X = master_df.to_numpy()
X = X.reshape(1, X.shape[0], X.shape[1])

# num_sim = 1
# team1 = "CINCINNATI"
# team2 = "XAVIER"
#
# predict_X_1 = get_game(team1, team2, location=0)
# predict_X_2 = get_game(team2, team1, location=2)
#
def simulate_game(team, predict_data):
    preds = []
    predict_X = predict_data.to_numpy()
    predict_X = np.reshape(predict_X, (1, predict_X.shape[0], predict_X.shape[1]))
    #predict_X = predict_X.reshape(predict_X.shape[1], 1, predict_X.shape[2])
    predictions = model.predict(predict_X)
    preds.append(predictions)
    df = pd.DataFrame([preds], columns=[team])
    return int(df[team].mean())
#
# output = team1 + ": " + str(simulate_game(team1, predict_X_1))
# output2 = team2 + ": " + str(simulate_game(team2, predict_X_2))
# print(output)
# print(output2)


teams_1, teams_2, is_neutral = get_scores_for_date(20211218)

results = []
score_results = []
for team1_obj, team2_obj, loc in zip(teams_1, teams_2, is_neutral):
    team1 = list(team1_obj.keys())[0]
    team2 = list(team2_obj.keys())[0]
    did_away_win_pred = False
    team1_score = list(team1_obj.values())[0]
    team2_score = list(team2_obj.values())[0]
    did_away_win = (team1_score == max([team1_score, team2_score]))
    try:
        if loc == 1:
            predict_X_1 = get_game(team1, team2, location=1)
            predict_X_2 = get_game(team2, team1, location=1)
        else:
            predict_X_1 = get_game(team1, team2, location=0)
            predict_X_2 = get_game(team2, team1, location=2)
        team1_pred = simulate_game(team1, predict_X_1)
        team2_pred = simulate_game(team2, predict_X_2)
        team1_dif = team1_score - team1_pred
        team2_dif = team2_score - team2_pred
        did_away_win_pred = (team1_pred == max([team1_pred, team2_pred]))
        print(team1 + ": " + str(team1_dif))
        print(team2 + ": " + str(team2_dif))
        result = (did_away_win == did_away_win_pred)
        print(result)
        results.append(result)
        score_results.append(abs(team1_dif))
        score_results.append(abs(team2_dif))
        print('\n')
    except ValueError as e:
        # this means the team could not be found, just skip it.
        pass
    except Exception as e:
        print("Failed " + team1, team2)
print(float(results.count(True)/len(results)))
print(str(statistics.mean((score_results))))


