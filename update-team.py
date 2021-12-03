import pandas as pd
from sportsipy.ncaab.teams import Teams
team_df = []
for team in Teams(2021):
    try:
        df = team.dataframe
        df = df.drop(columns=['conference', 'name', 'opp_offensive_rating', 'net_rating'])
        team_df.append(df)
    except Exception as e:
        print("FAILED" + str(team))
print(len(team_df))
df = pd.concat(team_df)
df.to_csv('teams2021.csv')