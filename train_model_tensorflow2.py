import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import train_test_split

columns_drop = ['away_losses', 'away_wins', 'conference_losses', 'conference_wins', 'games_played', 'home_losses',
                'home_wins', 'losses', 'wins',  'away_losses_2', 'away_wins_2', 'conference_losses_2', 'conference_wins_2',
                'games_played_2', 'home_losses_2', 'home_wins_2', 'losses_2', 'wins_2',]


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


master_df = pd.read_csv('team_list_scores2022.csv')
y = master_df['points_for'].to_numpy()
master_df = master_df.drop(columns=['abbreviation', 'abbreviation_2', 'Unnamed: 0.1','Unnamed: 0',  'Unnamed: 0_2', 'points_for', 'points_against'])
master_df = master_df.drop(columns=columns_drop)
#X = np.round(master_df.to_numpy(), 2).tolist()
X = np.round(master_df.to_numpy(), 2)

X_train, X_test, y_train, y_test = train_test_split(X, y)
model = tf.keras.Sequential([
    tf.keras.layers.Input(147),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(286, ),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(572),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

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

    tf.keras.layers.Dense(5000, ),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    #
    tf.keras.layers.Dense(1430, ),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    #
    # tf.keras.layers.Dense(572),
    # tf.keras.layers.PReLU(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(572,),
    tf.keras.layers.PReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1, ),
])

#model.compile(loss='mse', optimizer='adadelta')
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy', f1_metric])

model.fit(X_train, y_train, validation_data=(X_test,y_test), steps_per_epoch=100, epochs=4000)
json_file = model.to_json()
with open('dense.json', "w") as file:
    file.write(json_file)
model.save_weights('dense.h5')
print('model trained')

# output = team1 + ": " + str(simulate_game(team1, predict_X_1))
# output2 = team2 + ": " + str(simulate_game(team2, predict_X_2))
# print(output)
# print(output2)


