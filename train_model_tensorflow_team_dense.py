import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

columns_drop = ['away_losses', 'away_wins', 'conference_losses', 'conference_wins', 'games_played', 'home_losses',
                'home_wins', 'losses', 'wins',  'away_losses_2', 'away_wins_2', 'conference_losses_2', 'conference_wins_2',
                'games_played_2', 'home_losses_2', 'home_wins_2', 'losses_2', 'wins_2', 'minutes_played', 'minutes_played_2']

#team_list2018 = pd.read_csv('team_list_scores2018.csv').dropna()
team_list2019 = pd.read_csv('team_list_scores2019.csv').dropna()
#team_list2020 = pd.read_csv('team_list_scores2020.csv').dropna()
#team_list2021 = pd.read_csv('team_list_scores2021.csv').dropna()
team_list2022 = pd.read_csv('team_list_scores2022.csv').dropna()
master_df = pd.concat([team_list2022, team_list2019])
#                       team_list2020, team_list2019, team_list2018])
#master_df = pd.read_csv('team_list_scores2022.csv')
y = master_df['points_for'].to_numpy()
master_df = master_df.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2',
                                    'points_for', 'points_against'], errors='ignore')
master_df = master_df.drop(columns=columns_drop, errors='ignore')
X = master_df.to_numpy()
# scaler = MinMaxScaler()
# # fit scaler on data
# scaler.fit(X)
# # apply transform
# X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43)
model = tf.keras.Sequential([
    #tf.keras.layers.Input(183),

    tf.keras.layers.Dense(183, input_dim=183, ),
    #tf.keras.layers.PReLU(),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.Dropout(0.1),

    #tf.keras.layers.Dense(69, input_dim=69,),
    #tf.keras.layers.PReLU(),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.Dropout(0.1),

    # tf.keras.layers.Dense(512,),
    # tf.keras.layers.Dropout(0.1),
    # # tf.keras.layers.Dense(1000,),
    # # tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(85, ),
    # tf.keras.layers.Dense(1024,),
    # tf.keras.layers.Dense(4096, ),
    #
    #tf.keras.layers.Dense(512, kernel_regularizer=regularizers.l2(0.00001)),
    #tf.keras.layers.PReLU(),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.Dropout(0.1),

    # tf.keras.layers.Dense(1024),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.1),
    #
    #tf.keras.layers.Dense(64,),
    #tf.keras.layers.PReLU(),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.Dropout(0.1),

    # tf.keras.layers.Dense(10240,),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.1),
    #
    # tf.keras.layers.Dense(7000, activation='tanh'),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.1),

    # tf.keras.layers.Dense(572),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(1, kernel_initializer='normal'),
])

#model.compile(loss='mse', optimizer='adadelta')
#opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)
json_file = model.to_json()
with open('dense.json', "w") as file:
    file.write(json_file)
model.save_weights('dense.h5')
print('model trained')

# output = team1 + ": " + str(simulate_game(team1, predict_X_1))
# output2 = team2 + ": " + str(simulate_game(team2, predict_X_2))
# print(output)
# print(output2)
