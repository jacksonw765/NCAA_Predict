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

# cin = Schedule('Cincinnati', year='2021')
# bs_df = []
# for game in cin:
#    bs = game.dataframe['boxscore_index']

# data = Boxscoref('2021-09-04-cincinnati')
# df = data.dataframe
# df = df.drop(columns=['losing_abbr', 'losing_name', 'stadium', 'time', 'winner', 'winning_abbr','winning_name'])

df_list = []
for x in range(2017, 2021):
    for team in Teamsf(str(x)):
        df_list.append(team.dataframe)
df = pd.concat(df_list)
df = df.drop(columns=['abbreviation', 'conference', 'games', 'name', 'conference_losses', 'conference_win_percentage', 'conference_wins'])
df = df.dropna()
y = df['points_per_game'].astype(float).round(0).astype(int).to_numpy().tolist()
df = df.drop(columns=['points_per_game'])

X = df.to_numpy().tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y)
train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train = train.repeat().shuffle(1000).batch(32)
test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1)

#x_train = np.array(X_train)
#x_train = x_train.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = tf.keras.Sequential([
    tf.keras.Input((48,)),
    #tf.keras.layers.LSTM(48, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    #tf.keras.layers.LSTM(48, return_sequences=False),
    # tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(140,)),
    # #tf.keras.Input((48,)),
    # #tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 8), data_format="channels_first", activation="relu"),
    # #tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2500, activation=tf.nn.relu),
    tf.keras.layers.Dense(2500, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.softmax)
])

# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer=Adam(lr=0.0001),
#     metrics=['accuracy'])

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(train, validation_data=test, steps_per_epoch=100, epochs=30)

# get 2020 data
df_test = []
for team in Teamsf('2021'):
    df_test.append(team.dataframe)
df = pd.concat(df_test)
df = df.drop(columns=['abbreviation', 'conference', 'games', 'name', 'conference_losses', 'conference_win_percentage', 'conference_wins'])
df = df.dropna()
predict_true_labels = df['points_per_game'].astype(float).round(0).astype(int).to_numpy().tolist()
df = df.drop(columns=['points_per_game'])
predict_X = df.to_numpy().tolist()
predictions = model.predict(predict_X)

for pred_dict, expected in zip(predictions, predict_true_labels):
  predicted_index = pred_dict.argmax()
  probability = pred_dict.max()
  print(f"Prediction is {predicted_index} ({100 * probability:.1f}%), expected {expected}")