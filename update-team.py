import statistics
from kenpompy.utils import login
import kenpompy.summary as kp
from abbr_table import get_team_from_abbr
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

four_convert_to_int = ['AdjTempo', 'AdjOE', 'Off-eFG%', 'Off-TO%', 'Off-OR%', 'Off-FTRate',
       'AdjDE', 'Def-eFG%', 'Def-TO%', 'Def-OR%', 'Def-FTRate',]

height_convert_to_int = ['AvgHgt', 'EffHgt', 'C-Hgt', 'PF-Hgt', 'SF-Hgt', 'SG-Hgt', 'PG-Hgt',
       'Experience', 'Bench', 'Continuity']

year = 2022
browser = login('jacksonw765@gmail.com', '9797Jdw!')
kenpom_four_df = kp.get_fourfactors(browser, season=year)
kenpom_height_df = kp.get_height(browser, season=year)
kenpom_height_df['abbreviation'] = kenpom_height_df['Team'].str.upper()
kenpom_height_df['abbreviation'] = kenpom_height_df['abbreviation'].str.replace(' ', '-')

kenpom_height_df = kenpom_height_df[kenpom_height_df.columns.drop(list(kenpom_height_df.filter(regex='Rank')))]\
    .drop(columns=['Conference', 'Team'])
kenpom_height_df['abbreviation'] = kenpom_height_df['abbreviation'].map(get_team_from_abbr)
kenpom_height_df['abbreviation'] = kenpom_height_df['abbreviation'].str.replace('ST.', 'STATE')
kenpom_height_df['abbreviation'] = kenpom_height_df['abbreviation'].str.replace('STATETE', 'STATE')
kenpom_height_df['abbreviation'] = kenpom_height_df['abbreviation'].str.replace('.', '')
kenpom_four_df['abbreviation'] = kenpom_four_df['Team'].str.upper()
kenpom_four_df['abbreviation'] = kenpom_four_df['abbreviation'].str.replace(' ', '-')
kenpom_four_df = kenpom_four_df[kenpom_four_df.columns.drop(list(kenpom_four_df.filter(regex='Rank')))]\
    .drop(columns=['Conference', 'Team'])
kenpom_four_df['abbreviation'] = kenpom_four_df['abbreviation'].map(get_team_from_abbr)
kenpom_four_df['abbreviation'] = kenpom_four_df['abbreviation'].str.replace('ST.', 'STATE')
kenpom_four_df['abbreviation'] = kenpom_four_df['abbreviation'].str.replace('.', '')

for x in four_convert_to_int:
    kenpom_four_df[x] = pd.to_numeric(kenpom_four_df[x])

for x in height_convert_to_int:
    kenpom_height_df[x] = pd.to_numeric(kenpom_height_df[x])

for team in Teams(year):
    print(team)
    try:
        df = team.dataframe
        #player_heights = []
        #player_weight = []
        #for player in team.roster.players:
        #    try:
        #        height = str(player.dataframe['height'].to_list()[-2]).split('-')
        #        height = int(height[0]) * 12 + int(height[1])
        #        weight = int(player.dataframe['weight'].to_list()[-2])
        #        player_weight.append(weight)
        #        player_heights.append(height)
        #    except Exception as e:
        #        pass
        for x in columns_divide_by_zero:
            df[x] = df[x] / 100
        for x in columns_divide_by_minutes:
            df[x] = df[x] / df['minutes_played']
        df = df.drop(columns=['conference', 'name', 'opp_offensive_rating', 'net_rating',
                              'conference_wins', 'conference_losses', 'games_played', 'minutes_played'])
        #data = [statistics.mean(player_weight), statistics.mean(player_heights)]
        #df_player = pd.DataFrame([data], columns=['avg_weight', 'avg_height'])
        #df = df.reset_index(drop=True).merge(df_player.reset_index(drop=True), left_index=True, right_index=True)
        df = df.merge(kenpom_four_df, on='abbreviation', how='inner')
        df = df.merge(kenpom_height_df, on='abbreviation', how='inner')
        team_df.append(df)
    except Exception as e:
        print("FAILED" + str(team))
print(len(team_df))
df = pd.concat(team_df)
df.to_csv('teams{}.csv'.format(year))
