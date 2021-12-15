import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import train_test_split

columns_drop = ['away_losses', 'away_wins', 'conference_losses', 'conference_wins', 'games_played', 'home_losses',
                'home_wins', 'losses', 'wins',  'away_losses_2', 'away_wins_2', 'conference_losses_2', 'conference_wins_2',
                'games_played_2', 'home_losses_2', 'home_wins_2', 'losses_2', 'wins_2' 'team2_minutes_played', 'team1_minutes_played',
                'team1_losses', 'team2_losses', 'team1_win_percentage', 'team2_win_percentage', 'team2_minutes_played']


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


boxscores2020 = pd.read_csv('teams_list_boxscores2020.csv').dropna()
boxscores2021 = pd.read_csv('teams_list_boxscores2021.csv').dropna()
boxscores2022 = pd.read_csv('teams_list_boxscores2022.csv').dropna()
master_df = pd.concat([boxscores2021, boxscores2022, boxscores2020])
y = master_df['points_for'].to_numpy()
master_df = master_df.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2',
                                    'points_for', 'points_against', 'team1', 'team2'], errors='ignore')
master_df = master_df.drop(columns=columns_drop, errors='ignore')
#X = np.round(master_df.to_numpy(), 2).tolist()
X = master_df.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train = X_train.reshape(-1, X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(-1, X_test.shape[0], X_test.shape[1])
y_train = y_train.reshape(-1, y_train.shape[0])
y_test = y_test.reshape(1, y_test.shape[0],)
#X_train = X_train.reshape(-1, )
model = tf.keras.Sequential([


    tf.keras.layers.LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    #tf.keras.layers.SimpleRNN(512, return_sequences=True, activation='relu'),
    #tf.keras.layers.Dropout(0.1),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.LSTM(5, return_sequences=False),
    #tf.keras.layers.Dropout(0.1),

    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(286, ),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    #
    # tf.keras.layers.Dense(572),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.4),
    #
    # tf.keras.layers.Dense(1430, ),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),

    # tf.keras.layers.Dense(572,),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.4),
    #
    # tf.keras.layers.Dense(200, ),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    #
    # tf.keras.layers.Dense(100,),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    #
    # tf.keras.layers.Dense(50, ),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.4),

    # tf.keras.layers.Dense(2860, ),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    #
    # tf.keras.layers.Dense(1430, ),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    #
    # tf.keras.layers.Dense(572),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.4),

    # tf.keras.layers.Dense(286,),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1, ),
])

#model.compile(loss='mse', optimizer='adadelta')
model.compile(loss='mean_squared_error', optimizer='adam',)

model.fit(X_train, y_train, validation_data=(X_test,y_test), steps_per_epoch=100, epochs=1)
json_file = model.to_json()
with open('lstm.json', "w") as file:
    file.write(json_file)
model.save_weights('lstm.h5')
print('model trained')

# output = team1 + ": " + str(simulate_game(team1, predict_X_1))
# output2 = team2 + ": " + str(simulate_game(team2, predict_X_2))
# print(output)
# print(output2)


