import statistics

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBRegressor
from get_game_schedule import get_scores_for_date, get_schedules_for_date, get_favored_team_for_date

columns_drop = ['away_losses', 'away_wins', 'conference_losses', 'conference_wins', 'games_played', 'home_losses',
                'home_wins', 'losses', 'wins', 'away_losses_2', 'away_wins_2', 'conference_losses_2',
                'conference_wins_2',
                'games_played_2', 'home_losses_2', 'home_wins_2', 'losses_2', 'wins_2', 'minutes_played',
                'minutes_played_2', ]


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
    append_team = append_team.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0', 'Unnamed: 0_2'],
                                   errors='ignore')
    append_team = append_team.drop(columns=columns_drop, errors='ignore')
    location_data = [location_home, location_away, location_neutral]
    location_df = pd.DataFrame([location_data], columns=['location_home', 'location_away', 'location_neutral'])
    append_team = append_team.reset_index(drop=True).merge(location_df.reset_index(drop=True), left_index=True,
                                                           right_index=True)
    return append_team


# team_list2019 = pd.read_csv('team_list_scores2019.csv').dropna()
# team_list2021 = pd.read_csv('team_list_scores2021.csv')
# team_list2022 = pd.read_csv('team_list_scores2022.csv')
# master_df = pd.concat([team_list2022,team_list2019])
# master_df = pd.concat([team_list2022, team_list2021])
master_df = pd.read_csv('team_list_scores2022.csv')
y = master_df['points_for'].to_numpy()
master_df = master_df.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1', 'Unnamed: 0', 'Unnamed: 0_2',
                                    'points_for', 'points_against', 'Unnamed: 0', 'Unnamed: 0_2'], errors='ignore')
master_df = master_df.drop(columns=columns_drop, errors='ignore')
X = master_df.to_numpy()

# scaler = MinMaxScaler()
# # fit scaler on data
# scaler.fit(X)
# # apply transform
# X = scaler.transform(X)

num_sim = 1
estimators = 500
# EASTERN-KENTUCKY: 64
# SOUTHERN-CALIFORNIA: 86
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43)
model3 = XGBRegressor(n_estimators=estimators, learning_rate=0.01,
                      gamma=2,
                      max_depth=None,
                      min_child_weight=1,
                      colsample_bytree=0.5,
                      subsample=0.8,
                      reg_alpha=1,
                      # objective='reg:squarederror',
                      base_score=0)
model2 = LinearRegression(fit_intercept=False, positive=True)
#model = RandomForestRegressor(n_estimators=estimators)
#model.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

def simulate_game(team, predict_data, offset):
    preds = []
    for x in range(0, num_sim):
        #prediction1 = model.predict(predict_data.to_numpy())
        prediction2 = model2.predict(predict_data.to_numpy())
        prediction3 = model3.predict(predict_data.to_numpy())
        pred3 = np.round(prediction3[0] + offset, 0).tolist()
        #pred = np.round(prediction1[0] + offset, 0).tolist()
        pred2 = np.round(prediction2[0] + offset, 0).tolist()
        #preds.append(pred)
        preds.append(pred2)
        preds.append(pred3)
    df = pd.DataFrame(preds, columns=[team])
    return int(round(df[team].mean(), 0))


# team1 = "PURDUE"
# team2 = "ILLINOIS"
#
# predict_X_1 = get_game(team1, team2, location=0)
# predict_X_2 = get_game(team2, team1, location=2)
#
# output = team1 + ": " + str(simulate_game(team1, predict_X_1, 0))
# output2 = team2 + ": " + str(simulate_game(team2, predict_X_2, 0))
# print(output)
# print(output2)

# teams_1, teams_2 = get_schedules_for_date(20220226)

teams_1, teams_2 = get_scores_for_date(20220227)
winners, losers = get_favored_team_for_date('2022-02-27')
# teams_1_2, teams_2_2, is_neutral_2 = get_scores_for_date(20220114)
#
# teams_1 = teams_1 + teams_1_2
# teams_2 = teams_2 + teams_2_2
# is_neutral = is_neutral + is_neutral_2
#
#
score_results_home, score_results_away, score_results, results, score_results_home_ken, score_results_away_ken, = \
    [], [], [], [], [], []
for team1_obj, team2_obj in zip(teams_1, teams_2):
    team1 = list(team1_obj.keys())[0]
    team2 = list(team2_obj.keys())[0]
    did_away_win_pred = False
    team1_score = list(team1_obj.values())[0]
    team2_score = list(team2_obj.values())[0]
    did_away_win = (team1_score == max([team1_score, team2_score]))

    ken_pred_home_score = winners.get(team1, None)
    ken_pred_home_team = team1
    ken_pred_away_score = losers.get(team2, None)
    ken_pred_away_team = team2
    if not ken_pred_home_score:
        ken_pred_home_score = winners.get(team2)
        ken_pred_home_team = team2
        ken_pred_away_score = losers.get(team1, None)
        ken_pred_away_team = team1

    try:
        predict_X_1 = get_game(team1, team2, location=0)
        predict_X_2 = get_game(team2, team1, location=2)
        team1_pred = simulate_game(team1, predict_X_1, 0)
        team2_pred = simulate_game(team2, predict_X_2, 0)
        team1_dif = team1_score - team1_pred
        team2_dif = team2_score - team2_pred
        did_away_win_pred = (team1_pred == max([team1_pred, team2_pred]))
        if team1 == ken_pred_home_team:
            print(team1 + ": " + str(team1_pred) + ", " + str(ken_pred_home_score) + ", " + str(team1_score))
            #print(ken_pred_home_team + ": " + str(ken_pred_home_score))
            print(team2 + ": " + str(team2_pred) + ", " + str(ken_pred_away_score) + ", " + str(team2_score))
            #print(ken_pred_away_team + ": " + str(ken_pred_away_score))
        else:
            print(team1 + ": " + str(team1_pred) + ", " + str(ken_pred_home_score) + ", " + str(team1_score))
            #print(ken_pred_away_team + ": " + str(ken_pred_away_score))
            print(team2 + ": " + str(team2_pred) + ", " + str(ken_pred_away_score) + ", " + str(team2_score))
            #print(ken_pred_home_team + ": " + str(ken_pred_home_score))
        result = (did_away_win == did_away_win_pred)
        print(result)
        results.append(result)
        score_results_home.append(team1_dif)
        score_results_away.append(team2_dif)
        score_results.append(abs(team1_dif))
        score_results.append(abs(team2_dif))
        print('\n')
    except ValueError as e:
        # this means the team could not be found, just skip it.
        pass
    except Exception as e:
        print("Failed " + team1, team2)
print("% Correct: " + str(float(results.count(True) / len(results))))
print("Away: " + str(statistics.mean(score_results_away)))
print("Home: " + str(statistics.mean(score_results_home)))
print(str(statistics.mean(score_results)))
print("Predicted " + str(len(score_results_home)))

# for team1, team2 in zip(teams_1, teams_2):
#     pred_win_score = winners.get(team1, None)
#     pred_win_team = team1
#     pred_loss_score = losers.get(team2, None)
#     pred_loss_team = team2
#     if not pred_win_score:
#         pred_win_score = winners.get(team2)
#         pred_win_team = team2
#         pred_loss_score = losers.get(team1, None)
#         pred_loss_team = team1
#     try:
#         predict_X_1 = get_game(team1, team2, location=0)
#         predict_X_2 = get_game(team2, team1, location=2)
#         team1_pred = simulate_game(team1, predict_X_1, 0)
#         team2_pred = simulate_game(team2, predict_X_2, 0)
#         print('Pred')
#         if pred_win_team == team1:
#             print(pred_win_team + ': ' + str(pred_win_score))
#             print(pred_loss_team + ': ' + str(pred_loss_score))
#         else:
#             print(pred_loss_team + ': ' + str(pred_loss_score))
#             print(pred_win_team + ': ' + str(pred_win_score))
#         print('Us')
#         print(team1 + ': ' + str(team1_pred))
#         print(team2 + ': ' + str(team2_pred))
#         print('\n')
#
#     except ValueError as e:
#         # this means the team could not be found, just skip it.
#         #print("Failed " + team1, team2)
#         pass
#     except Exception as e:
#         print("Failed " + team1, team2)
