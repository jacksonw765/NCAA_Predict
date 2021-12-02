import pandas as pd
from keras.optimizer_v2.adam import Adam
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

columns_drop = ['away_losses', 'away_wins', 'conference_losses', 'conference_wins', 'games_played', 'home_losses',
                'home_wins', 'losses', 'wins',  'away_losses_2', 'away_wins_2', 'conference_losses_2', 'conference_wins_2',
                'games_played_2', 'home_losses_2', 'home_wins_2', 'losses_2', 'wins_2']


def get_game(team1, team2):
    df_teams = pd.read_csv('teams.csv')
    df_teams_columns = df_teams.columns.tolist()
    df_teams_columns_new = []
    for x in df_teams_columns:
        df_teams_columns_new.append(x + "_2")
    team_1 = df_teams[df_teams['abbreviation'] == team1]
    team_2 = df_teams[df_teams['abbreviation'] == team2]
    team_2.columns = df_teams_columns_new
    append_team = team_1.reset_index(drop=True).merge(team_2.reset_index(drop=True),
                                                         left_index=True, right_index=True)
    append_team = append_team.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0', 'Unnamed: 0_2'])
    append_team = append_team.drop(columns=columns_drop)
    append_team = append_team.filter(regex='percentage')
    return append_team


master_df = pd.read_csv('team_list_scores.csv')
y = master_df[['points_for', 'points_against']].to_numpy().tolist()
master_df = master_df.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2', 'points_for', 'points_against'])
master_df = master_df.filter(regex='percentage')
master_df = master_df.drop(columns=columns_drop)
X = np.round(master_df.to_numpy(), 3).tolist()
#X = np.expand_dims(X, axis=0)
#y = np.expand_dims(y, axis=0)
#y =
#X = np.reshape(X,(X.shape[0], 1, X.shape[1]))
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LinearRegression()
results = model.fit(X_train, y_train)
# train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# train = train.repeat().shuffle(1000).batch(32)
# test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1)
#
# X = np.expand_dims(X, axis=0)
#
# model = tf.keras.Sequential([
#     #tf.keras.Input((50,)),
#     #tf.keras.Conv2D(20, (5, 5), )
#     tf.keras.layers.LSTM(48, return_sequences=True, input_shape=X),
#     #tf.keras.layers.Conv2D(activation="relu", kernel_size = 32, filters = 32),
#     #tf.keras.layers.Flatten(),
#     #tf.keras.layers.Dense(2500, activation=tf.nn.relu),
#     #tf.keras.layers.Dense(units=50, activation=tf.nn.relu, input_shape=(50,)),
#     tf.keras.layers.Dense(2500, activation=tf.nn.relu),
#     tf.keras.layers.Dense(2500, activation=tf.nn.relu),
#     #tf.keras.layers.Dense(10000, activation=tf.nn.relu),
#     tf.keras.layers.Dense(2, activation=tf.nn.softmax)
# ])
#
# model.compile(
#     loss='mean_squared_error',
#     optimizer='adam',
#     metrics=['accuracy'])
#
# model.fit(train, validation_data=test, steps_per_epoch=100, epochs=20, batch_size=10, verbose=True)
predict_X = get_game('MIAMI-OH', "CINCINNATI")
predictions = model.predict(predict_X)
np.round(predictions)
#for pred in predictions:
#    for pred2 in pred:
#        print(pred2*100)