import pandas as pd
from sportsipy.ncaab.teams import Teams

columns_divide_by_minutes = ["assists", "blocks", "defensive_rebounds", "field_goal_attempts", "field_goals",
                   "free_throw_attempts", "offensive_rebounds", "opp_assists", "opp_blocks", "opp_defensive_rebounds",
                   "opp_field_goal_attempts", "opp_field_goals", "opp_free_throw_attempts", "opp_free_throws",
                   "opp_offensive_rebounds", "opp_personal_fouls", "opp_points", "opp_steals",
                   "opp_three_point_field_goal_attempts", "opp_three_point_field_goals",
                   "opp_two_point_field_goal_attempts", "opp_two_point_field_goals", "opp_total_rebounds",
                   "opp_turnovers", 'free_throws',
                   "personal_fouls", "points", "steals", "three_point_field_goal_attempts", "three_point_field_goals",
                   "two_point_field_goal_attempts", "two_point_field_goals", "total_rebounds", "turnovers"]

columns_divide_by_zero = ['assist_percentage', 'block_percentage', 'steal_percentage', 'offensive_rebound_percentage',
                          'turnover_percentage', 'total_rebound_percentage', 'opp_assist_percentage', 'opp_block_percentage',
                          'opp_total_rebound_percentage', 'opp_turnover_percentage', 'opp_offensive_rebound_percentage',
                          'opp_steal_percentage']

player_lists = []
for team in Teams(2022):
    #team_name = str(team).split('(')[0]
    for player in team.roster.players:
        try:
            df = player.dataframe.iloc[-1].to_frame().transpose()
            player_id = df['player_id'].values[0]
            df = df.rename({'Career': player_id}, axis=0)
            height = df['height'].values[0].split('-')
            height = (int(height[0])*12) + int(height[1])
            df = df.drop(columns=['position', 'weight', 'player_id', 'conference'])
            #df['team'] = team_name
            df['height'] = height
            player_lists.append(df)
        except Exception as e:
            print("error: " + str(player))
print(len(player_lists))
df = pd.concat(player_lists)
df.to_csv('players.csv')