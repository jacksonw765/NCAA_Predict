import pandas as pd
#import xgboost as xgb
#from sklearn import preprocessing
#from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import *
#from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np

#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

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


master_df = pd.read_csv('team_list_scores2021.csv')
y = master_df['points_for'].to_numpy().tolist()
master_df = master_df.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2', 'points_for', 'points_against'])
master_df = master_df.drop(columns=columns_drop)
X = np.round(master_df.to_numpy(), 2).tolist()

num_sim = 1
team1 = "ALABAMA"
team2 = "GONZAGA"

predict_X_1 = get_game(team1, team2, location=0)
predict_X_2 = get_game(team2, team1, location=2)

def simulate_game(team, predict_data):
    preds = []
    for x in range(0, num_sim):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        #model = LogisticRegression(max_iter=10000, penalty="l2", fit_intercept=False, multi_class="ovr", C=1, solver = 'lbfgs')
        #model = RandomForestRegressor(n_estimators=200)
        model = LinearRegression()
        model.fit(X_train, y_train)
        print(model.score(X_train, y_train))
        predictions = model.predict(predict_data.to_numpy())
        pred = np.round(predictions, 0).tolist()
        preds.append(pred)
    df = pd.DataFrame(preds[0], columns=[team])
    return int(round(df[team].mean(), 0))


print(team1 + ": " + str(simulate_game(team1, predict_X_1)))
print(team2 + ": " + str(simulate_game(team2, predict_X_2)))

