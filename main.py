import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.applications.densenet import layers
from keras.optimizer_v2.adam import Adam
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
    return append_team.filter(regex='percentage')


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


master_df = pd.read_csv('team_list_scores2020.csv')
y = master_df['points_for'].to_numpy().tolist()
master_df = master_df.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2', 'points_for', 'points_against'])
master_df = master_df.drop(columns=columns_drop).filter(regex='percentage')
#X = np.round(master_df.to_numpy(), 2).tolist()
X = master_df.to_numpy().tolist()

num_sim = 1
team1 = "ALABAMA"
team2 = "GONZAGA"

predict_X_1 = get_game(team1, team2, location=0)
predict_X_2 = get_game(team2, team1, location=2)

def simulate_game(team, predict_data):
    preds = []
    for x in range(0, num_sim):

        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Input(50),
        #
        #     tf.keras.layers.Dense(286, ),
        #     tf.keras.layers.PReLU(),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.2),
        #
        #     tf.keras.layers.Dense(572),
        #     tf.keras.layers.PReLU(),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.4),
        #
        #     tf.keras.layers.Dense(1430, ),
        #     tf.keras.layers.PReLU(),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.2),
        #
        #     tf.keras.layers.Dense(2860, ),
        #     tf.keras.layers.PReLU(),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.2),
        #
        #     tf.keras.layers.Dense(5720, ),
        #     tf.keras.layers.PReLU(),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.2),
        #
        #     tf.keras.layers.Dense(2860, ),
        #     tf.keras.layers.PReLU(),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.2),
        #
        #     tf.keras.layers.Dense(1430, ),
        #     tf.keras.layers.PReLU(),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.2),
        #
        #     tf.keras.layers.Dense(572),
        #     tf.keras.layers.PReLU(),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.4),
        #
        #     tf.keras.layers.Dense(286,),
        #     tf.keras.layers.PReLU(),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.2),
        #
        #     tf.keras.layers.Dense(1, ),
        # ])

        # Model compiling settings
        #model.compile(loss='mse', optimizer='adadelta')
        #model.compile(loss='mse', optimizer='adam', metrics=['accuracy', f1_metric])

        #model.fit(X_train, y_train, validation_data=X_test, steps_per_epoch=100, epochs=2048)
        #model.save('model.h5')
        model = tf.keras.models.load_model('model.h5')
        predict_X = predict_data.to_numpy()
        predictions = model.predict([predict_X])
        # for pred_dict, expected in zip(predictions, predict_true_labels):
        #     predicted_index = pred_dict.argmax()
        #     probability = pred_dict.max()
        #     print(f"Prediction is {predicted_index} ({100 * probability:.1f}%), expected {expected}")
        preds.append(predictions)
    df = pd.DataFrame([preds], columns=[team])
    return int(df[team].mean())

output = team1 + ": " + str(simulate_game(team1, predict_X_1))
output2 = team2 + ": " + str(simulate_game(team2, predict_X_2))
print(output)
print(output2)


