import statistics

import pandas as pd
from sportsipy.ncaab.teams import Teams
team_df = []
year = 2022
for team in Teams(year):
    try:
        df = team.dataframe
        player_heights = []
        player_weight = []
        for player in team.roster.players:
            try:
                height = str(player.dataframe['height'].to_list()[-2]).split('-')
                height = int(height[0])*12 + int(height[1])
                weight = int(player.dataframe['weight'].to_list()[-2])
                player_weight.append(weight)
                player_heights.append(height)
            except Exception as e:
                pass
        df = df.drop(columns=['conference', 'name', 'opp_offensive_rating', 'net_rating'])
        data = [statistics.mean(player_weight), statistics.mean(player_heights)]
        df_player = pd.DataFrame([data], columns=['avg_weight', 'avg_height'])
        df = df.reset_index(drop=True).merge(df_player.reset_index(drop=True), left_index=True, right_index=True)
        team_df.append(df)
    except Exception as e:
        print("FAILED" + str(team))
print(len(team_df))
df = pd.concat(team_df)
df.to_csv('teams_weights{}.csv'.format(year))