import pandas as pd
import numpy as np
import tensorflow as tf
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
    return append_team


master_df = pd.read_csv('team_list_scores2021.csv')
y = master_df['points_for'].to_numpy().tolist()
master_df = master_df.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2', 'points_for', 'points_against'])
master_df = master_df.drop(columns=columns_drop)
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

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train = train.repeat().shuffle(1000).batch(32)
        test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1)
        model = tf.keras.Sequential([
            #tf.keras.Input((143,)),
            tf.keras.layers.LSTM(143, return_sequences=True, input_shape=(1, 143, 1)),
            # tf.keras.layers.LSTM(48, return_sequences=False),
            # tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(140,)),
            # #tf.keras.Input((48,)),
            # #tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 8), data_format="channels_first", activation="relu"),
            # #tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(5000, activation=tf.nn.relu),
            tf.keras.layers.Dense(5000, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.softmax)
        ])
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-05, amsgrad=False,
            name='Adam')

        # Model compiling settings
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

        model.fit(X_train, y_train, validation_data=X_test, steps_per_epoch=100, epochs=2)
        predict_X = predict_data.to_numpy().tolist()
        predictions = model.predict(predict_X)
        # for pred_dict, expected in zip(predictions, predict_true_labels):
        #     predicted_index = pred_dict.argmax()
        #     probability = pred_dict.max()
        #     print(f"Prediction is {predicted_index} ({100 * probability:.1f}%), expected {expected}")
        preds.append(predictions)
    df = pd.DataFrame(preds, columns=[team])
    return int(round(df[team].mean(), 0))

print(team1 + ": " + str(simulate_game(team1, predict_X_1)))
print(team2 + ": " + str(simulate_game(team2, predict_X_2)))


