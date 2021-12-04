import pandas as pd
from sklearn.model_selection import train_test_split
from sportsipy.ncaab.teams import Teams
from sportsipy.ncaaf.teams import Teams as Teamsf
from sklearn.datasets import load_iris
from sportsipy.ncaab.schedule import Schedule
from sportsipy.ncaab.boxscore import Boxscore


cin = Schedule('CINCINNATI', year='2022').dataframe['boxscore_index'].to_numpy().tolist()[0]
score = Boxscore(cin).dataframe
print(score)
# bs_df = []
# for game in cin:
#     index = game.dataframe['boxscore_index'].to_numpy().tolist()[0]
#     data = Boxscore(index).dataframe
#     print(data)