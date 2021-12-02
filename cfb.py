import pandas as pd
#import tensorflow as tf
#from sklearn.model_selection import train_test_split
from sportsipy.ncaab.teams import Teams
from sportsipy.ncaaf.teams import Teams as Teamsf
from sklearn.datasets import load_iris
from sportsipy.ncaab.schedule import Schedule
from sportsipy.ncaaf.boxscore import Boxscore as Boxscore

#cin = Schedule('Cincinnati', year='2022')
# bs_df = []
# for game in cin:
#     index = game.dataframe['boxscore_index'].to_numpy().tolist()[0]
#     data = Boxscore(index).dataframe
#     print(data)
# player_lists = []
# for team in Teams(2022):
#     team_name = str(team).split('(')[0]
#     for player in team.roster.players:
#         try:
#             df = player.dataframe.iloc[-1].to_frame().transpose()
#             player_id = df['player_id'].values[0]
#             df = df.rename({'Career': player_id}, axis=0)
#             height = df['height'].values[0].split('-')
#             height = (int(height[0])*12) + int(height[1])
#             df = df.drop(columns=['position', 'weight', 'player_id', 'height'])
#             df['team'] = team_name
#             df['height'] = height
#             player_lists.append(df)
#         except Exception as e:
#             print("error: " + str(player))
# print(len(player_lists))
# df = pd.concat(player_lists)
# print(df)

df_players = pd.read_csv('players.csv')
df_players_columns = df_players.columns.tolist()
df_players_columns_new = []
for x in df_players_columns:
    df_players_columns_new.append(x + "_2")
df_players = df_players.dropna()
all_teams = list(set(df_players['team'].to_numpy().tolist()))
df_sch_list = []
for team in all_teams:
    fixed = str(team.strip()).replace(' ', '-').replace('&', '').replace('.', '').replace("'", "").lower()
    try:
        sch = Schedule(fixed, year='2022')
        df_sch = sch.dataframe.dropna(subset=['boxscore_index'])
        team_opps = df_players['team'].to_numpy().tolist()
        for opponent in team_opps:
            opp_df = df_players[df_players['team'].str.contains(team)]
        #df_sch_list.append(df_sch)
    except Exception as e:
        print(fixed)
#df_sch = pd.concat(df_sch_list)

for team in all_teams:
    df_team = df_players[df_players['team'].str.contains(team)]
#df_cin = df[df['team'].str.contains('Cincinnati')]
#df_miami = df[df['team'].str.contains('miami oh')]

