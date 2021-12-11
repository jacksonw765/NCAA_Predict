import statistics

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import f1_score
from get_game_schedule import get_schedules_for_date
from xgboost import XGBRegressor
import xgboost as xgb

columns_drop = ['away_losses', 'away_wins', 'conference_losses', 'conference_wins', 'games_played', 'home_losses',
                'home_wins', 'losses', 'wins',  'away_losses_2', 'away_wins_2', 'conference_losses_2', 'conference_wins_2',
                'games_played_2', 'home_losses_2', 'home_wins_2', 'losses_2', 'wins_2', 'minutes_played', 'minutes_played_2']


def get_game(home_team, away_team, location=2):
    df_teams = pd.read_csv('teams2022.csv')
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

#master_df_w = pd.read_csv('team_list_scores2022_w.csv')
master_df = pd.read_csv('team_list_scores2022.csv')
y = master_df['points_for'].to_numpy()
master_df = master_df.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2', 'points_for', 'points_against'])
master_df = master_df.drop(columns=columns_drop)
X = np.round(master_df.to_numpy(), 2)

num_sim = 1500
# EASTERN-KENTUCKY: 64
# SOUTHERN-CALIFORNIA: 86

def simulate_game(team, predict_data):
    preds = []
    for x in range(0, num_sim):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        #model = LogisticRegression(max_iter=9000, penalty="l2", fit_intercept=False, multi_class="ovr", C=1, solver = 'lbfgs')
        #model = RandomForestRegressor(n_estimators=300)

        #clf = xgb.XGBClassifier(seed=2)
        #model = GridSearchCV(clf, param_grid=parameters, cv=5)
        model = LinearRegression(fit_intercept=False, positive=True)
        # model = XGBRegressor(n_estimators=1000, learning_rate=0.02,
        #                      gamma=2,
        #                      max_depth=None,
        #                      min_child_weight=1,
        #                      colsample_bytree=0.5,
        #                      subsample=0.8,
        #                      reg_alpha=1,
        #                      objective='reg:squarederror',
        #                      base_score=7.76
        #                      )

        # 0.6666666666666666
        # 6.666666666666667

        model = model.fit(X_train, y_train)
        #clf = model.best_estimator_
        #print(clf)
        #print(model.score(X_train, y_train))
        predictions = model.predict(predict_data.to_numpy())
        pred = np.round(predictions, 0).tolist()
        preds.append(pred)
    df = pd.DataFrame(preds[0], columns=[team])
    return int(round(df[team].mean(), 0))


team1 = "CINCINNATI"
team2 = "XAVIER"

predict_X_1 = get_game(team1, team2, location=0)
predict_X_2 = get_game(team2, team1, location=2)

print(team1 + ": " + str(simulate_game(team1, predict_X_1)))
print(team2 + ": " + str(simulate_game(team2, predict_X_2)))

# teams_1 = [{"APPALACHIAN-STATE": 65}, {'VERMONT': 70}, {"MURRAY-STATE": 74}, {"LOYOLA-IL": 69}, {'DEPAUL': 62}]
# teams_2 = [{"FURMAN": 73}, {'BROWN': 65}, {"MEMPHIS": 72}, {'MEMPHIS': 72}, {'LOUISVILLE': 55}]
#
# results = []
# score_results = []
# for team1_obj, team2_obj in zip(teams_1, teams_2):
#     team1 = list(team1_obj.keys())[0]
#     team2 = list(team2_obj.keys())[0]
#     did_away_win = False
#     did_away_win_pred = False
#     team1_score = list(team1_obj.values())[0]
#     team2_score = list(team2_obj.values())[0]
#     if team1_score > team2_score:
#         did_away_win = True
#     try:
#         predict_X_1 = get_game(team1, team2, location=0)
#         predict_X_2 = get_game(team2, team1, location=2)
#         team1_pred = simulate_game(team1, predict_X_1)
#         team2_pred = simulate_game(team2, predict_X_2)
#         team1_dif = team1_score - team1_pred
#         team2_dif = team2_score - team2_pred
#         if team1_pred > team2_pred:
#             did_away_win = True
#         print(team1 + ": " + str(team1_dif))
#         print(team2 + ": " + str(team2_dif))
#         result = (did_away_win == did_away_win_pred)
#         print(result)
#         results.append(result)
#         score_results.append(abs(team1_dif))
#         score_results.append(abs(team2_dif))
#         print('\n')
#     except Exception as e:
#         print("Failed " + team1, team2)
# print(float(results.count(True)/len(results)))
# print(str(statistics.mean((score_results))))