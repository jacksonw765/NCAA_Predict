import pandas as pd
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split

import numpy as np

columns_drop = ['away_losses', 'away_wins', 'conference_losses', 'conference_wins', 'games_played', 'home_losses',
                'home_wins', 'losses', 'wins',  'away_losses_2', 'away_wins_2', 'conference_losses_2', 'conference_wins_2',
                'games_played_2', 'home_losses_2', 'home_wins_2', 'losses_2', 'wins_2',]


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


master_df = pd.read_csv('teams_list_boxscores.csv').dropna()
y = master_df[['points_for', 'points_against']].to_numpy().tolist()
master_df = master_df.drop(columns=['Unnamed: 0', 'points_for', 'points_against', 'team1', 'team2'], errors='ignore')
#master_df = master_df.filter(regex='percentage')
X = np.round(master_df.to_numpy(), 2).tolist()
preds = []
away_team = "BRYANT"
home_team = 'CINCINNATI'
predict_X = get_game(home_team, away_team)
for x in range(0, 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = LinearRegression()
    results = model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
    predictions = model.predict(np.round(predict_X.to_numpy(), 2))
    pred = np.round(predictions, 0).tolist()
    preds.append(pred[0])

df = pd.DataFrame(preds, columns=[home_team, away_team])
print(home_team + ": " + str(round(df[home_team].mean(), 0)))
print(away_team + ": " + str(round(df[away_team].mean(),0)))