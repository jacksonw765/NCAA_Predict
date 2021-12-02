import pandas as pd
from sportsipy.ncaab.teams import Teams
team_df = []
for team in Teams(2022):
    #team_name = str(team).split('(')[0]
    df = team.dataframe
    team_df.append(df)
    # for player in team.roster.players:
    #     try:
    #         df = player.dataframe.iloc[-1].to_frame().transpose()
    #         player_id = df['player_id'].values[0]
    #         df = df.rename({'Career': player_id}, axis=0)
    #         height = df['height'].values[0].split('-')
    #         height = (int(height[0])*12) + int(height[1])
    #         df = df.drop(columns=['position', 'weight', 'player_id', 'height'])
    #         #df['team'] = team_name
    #         df['height'] = height
    #         player_lists.append(df)
    #     except Exception as e:
    #         print("error: " + str(player))
print(len(team_df))
df = pd.concat(team_df)
df.to_csv('teams.csv')