import statistics

import pandas as pd
from sklearn.linear_model import *
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import xgboost as xgb

from get_game_schedule import get_scores_for_date

columns_drop = ['away_losses', 'away_wins', 'conference_losses', 'conference_wins', 'games_played', 'home_losses',
                'home_wins', 'losses', 'wins',  'away_losses_2', 'away_wins_2', 'conference_losses_2', 'conference_wins_2',
                'games_played_2', 'home_losses_2', 'home_wins_2', 'losses_2', 'wins_2' 'team2_minutes_played', 'team1_minutes_played',
                'team1_losses', 'team2_losses', 'team1_win_percentage', 'team2_win_percentage', 'team2_minutes_played']


def get_game(home_team, away_team, location=2):
    df_teams = pd.read_csv('teams_list_boxscores.csv')
    team_1 = df_teams[df_teams['team1'] == home_team].filter(regex='team1').drop(columns=['team1'])
    team_2 = df_teams[df_teams['team2'] == away_team].filter(regex='team1').drop(columns=['team1'])
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
    return append_team

num_sim = 1

def simulate_game(team, predict_data):
    preds = []
    for x in range(0, num_sim):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        #model = Lasso()
        model = XGBRegressor(n_estimators=500, learning_rate=0.02,
                             gamma=2,
                             max_depth=None,
                             min_child_weight=1,
                             colsample_bytree=0.5,
                             subsample=0.8,
                             reg_alpha=1,
                             objective='reg:squarederror',
                             base_score=.05)

        model = model.fit(X_train, y_train)
        #print(model.score(X_train, y_train))
        predictions = model.predict(predict_data.to_numpy())
        pred = np.round(predictions, 0).tolist()
        preds.append(pred)
    df = pd.DataFrame(preds[0], columns=[team])
    return int(round(df[team].mean(), 0))


boxscores2020 = pd.read_csv('teams_list_boxscores2020.csv').dropna()
boxscores2021 = pd.read_csv('teams_list_boxscores2021.csv').dropna()
boxscores2022 = pd.read_csv('teams_list_boxscores2022.csv').dropna()
master_df = pd.concat([boxscores2021, boxscores2022, boxscores2020])
y = master_df[['points_for']].to_numpy()
master_df = master_df.drop(columns=['Unnamed: 0', 'points_for', 'points_against', 'team1', 'team2'], errors='ignore')
#master_df = master_df.filter(regex='percentage')
X = np.round(master_df.to_numpy(), 2).tolist()

# team1 = "CINCINNATI"
# team2 = "MARYLAND"
#
# predict_X_1 = get_game(team1, team2, location=0)
# predict_X_2 = get_game(team2, team1, location=2)
#
# print(team1 + ": " + str(simulate_game(team1, predict_X_1)))
# print(team2 + ": " + str(simulate_game(team2, predict_X_2)))

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