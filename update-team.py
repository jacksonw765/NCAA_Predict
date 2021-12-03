import pandas as pd
from sportsipy.ncaab.teams import Teams
team_df = []
for team in Teams(2022):
    df = team.dataframe
    df = df.drop(columns=['conference', 'name', 'opp_offensive_rating', 'net_rating'])
    team_df.append(df)
print(len(team_df))
df = pd.concat(team_df)
df.to_csv('teams2.csv')