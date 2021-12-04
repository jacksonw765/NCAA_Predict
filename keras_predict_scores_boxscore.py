import pandas as pd
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split

import numpy as np

columns_drop = ['away_losses', 'away_wins', 'conference_losses', 'conference_wins', 'games_played', 'home_losses',
                'home_wins', 'losses', 'wins',  'away_losses_2', 'away_wins_2', 'conference_losses_2', 'conference_wins_2',
                'games_played_2', 'home_losses_2', 'home_wins_2', 'losses_2', 'wins_2',]


def get_game(home_team, away_team, location=2):
    df_teams = pd.read_csv('teams2.csv')
    df_teams_columns_new = []
    #location = pd.DataFrame([2, 0], columns=['location'])
    team_1 = df_teams[df_teams['abbreviation'] == home_team]
    team_2 = df_teams[df_teams['abbreviation'] == away_team]
    team_2.columns = df_teams_columns_new
    append_team = team_1.reset_index(drop=True).merge(team_2.reset_index(drop=True), left_index=True, right_index=True)
    #append_team = append_team.filter(regex='percentage')
    append_team["location"] = [location]
    append_team = append_team.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0', 'Unnamed: 0_2'])
    append_team = append_team.drop(columns=columns_drop)
    return append_team


master_df = pd.read_csv('team_list_scores2.csv')
y = master_df[['points_for', 'points_against']].to_numpy().tolist()
master_df = master_df.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2', 'points_for', 'points_against'])
#master_df = master_df.filter(regex='percentage')
master_df = master_df.drop(columns=columns_drop)
X = np.round(master_df.to_numpy(), 2).tolist()
preds = []
away_team = "CINCINNATI"
home_team = 'MIAMI-OH'
predict_X = get_game(home_team, away_team)
for x in range(0, 100):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = LinearRegression()
    results = model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
    predictions = model.predict(predict_X.to_numpy())
    pred = np.round(predictions, 0).tolist()
    preds.append(pred[0])

df = pd.DataFrame(preds, columns=[home_team, away_team])
print(home_team + ": " + str(df[home_team].mean()))
print(away_team + ": " + str(df[away_team].mean()))