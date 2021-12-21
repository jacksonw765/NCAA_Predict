import statistics

import pandas as pd
from sportsipy.ncaab.teams import Teams

team_df = []
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
year = 2022
for team in Teams(year):
    print(team)
    try:
        df = team.dataframe
        player_heights = []
        player_weight = []
        for player in team.roster.players:
            try:
                height = str(player.dataframe['height'].to_list()[-2]).split('-')
                height = int(height[0]) * 12 + int(height[1])
                weight = int(player.dataframe['weight'].to_list()[-2])
                player_weight.append(weight)
                player_heights.append(height)
            except Exception as e:
                pass
        for x in columns_divide_by_zero:
            df[x] = df[x] / 100
        for x in columns_divide_by_minutes:
            df[x] = df[x] / df['minutes_played']
        df = df.drop(columns=['conference', 'name', 'opp_offensive_rating', 'net_rating',
                              'conference_wins', 'conference_losses', 'games_played', 'minutes_played'])
        data = [statistics.mean(player_weight), statistics.mean(player_heights)]
        df_player = pd.DataFrame([data], columns=['avg_weight', 'avg_height'])
        df = df.reset_index(drop=True).merge(df_player.reset_index(drop=True), left_index=True, right_index=True)
        team_df.append(df)
    except Exception as e:
        print("FAILED" + str(team))
print(len(team_df))
df = pd.concat(team_df)
df.to_csv('teams{}.csv'.format(year))
