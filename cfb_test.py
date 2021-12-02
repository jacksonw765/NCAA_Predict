import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sportsipy.ncaab.teams import Teams
from sportsipy.ncaaf.teams import Teams as Teamsf
from sklearn.datasets import load_iris
from sportsipy.ncaaf.schedule import Schedule
from sportsipy.ncaaf.boxscore import Boxscore as Boxscore

cin = Schedule('Miami-OH', year='2022')
bs_df = []
for game in cin:
    index = game.dataframe['boxscore_index'].to_numpy().tolist()[0]
    data = Boxscore(index).dataframe
    print(data)