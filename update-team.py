import statistics
from kenpompy.utils import login
import kenpompy.summary as kp
from abbr_table import get_team_from_abbr
import pandas as pd
from sportsipy.ncaab.teams import Teams

year = 2022
browser = login('jacksonw765@gmail.com', '9797Jdw!')

team_df = []
columns_divide_by_minutes = ["assists", "blocks", "defensive_rebounds", "field_goal_attempts", "field_goals",
                             "free_throw_attempts", "offensive_rebounds", "opp_assists", "opp_blocks",
                             "opp_defensive_rebounds",
                             "opp_field_goal_attempts", "opp_field_goals", "opp_free_throw_attempts", "opp_free_throws",
                             "opp_offensive_rebounds", "opp_personal_fouls", "opp_points", "opp_steals",
                             "opp_three_point_field_goal_attempts", "opp_three_point_field_goals",
                             "opp_two_point_field_goal_attempts", "opp_two_point_field_goals", "opp_total_rebounds",
                             "opp_turnovers", 'free_throws',
                             "personal_fouls", "points", "steals", "three_point_field_goal_attempts",
                             "three_point_field_goals",
                             "two_point_field_goal_attempts", "two_point_field_goals", "total_rebounds", "turnovers"]

columns_divide_by_hundo = ['assist_percentage', 'block_percentage', 'steal_percentage', 'offensive_rebound_percentage',
                           'turnover_percentage', 'total_rebound_percentage', 'opp_assist_percentage',
                           'opp_block_percentage',
                           'opp_total_rebound_percentage', 'opp_turnover_percentage',
                           'opp_offensive_rebound_percentage',
                           'opp_steal_percentage', 'AvgHgt', 'EffHgt', 'C-Hgt', 'PF-Hgt', 'SF-Hgt', 'SG-Hgt', 'PG-Hgt',
                           'Experience', 'Bench', 'Continuity', 'AdjTempo', 'AdjOE', 'Off-eFG%', 'Off-TO%', 'Off-OR%',
                           'Off-FTRate', 'AdjDE', 'Def-eFG%', 'Def-TO%', 'Def-OR%', 'Def-FTRate', ]

four_convert_to_int = ['AdjTempo', 'AdjOE', 'Off-eFG%', 'Off-TO%', 'Off-OR%', 'Off-FTRate',
                       'AdjDE', 'Def-eFG%', 'Def-TO%', 'Def-OR%', 'Def-FTRate', ]

height_convert_to_int = ['AvgHgt', 'EffHgt', 'C-Hgt', 'PF-Hgt', 'SF-Hgt', 'SG-Hgt', 'PG-Hgt',
                         'Experience', 'Bench', 'Continuity']

point_convert_to_int = ['Off-FT', 'Off-2P', 'Off-3P', 'Def-FT', 'Def-2P', 'Def-3P',]

eff_convert_to_int = ['Tempo-Adj', 'Avg. Poss Length-Offense', 'Avg. Poss Length-Defense',
       'Off. Efficiency-Adj', 'Def. Efficiency-Adj',]

def clean_ken(df):
    df['abbreviation'] = df['Team'].str.upper()
    df['abbreviation'] = df['abbreviation'].str.replace(' ', '-')
    df = df[df.columns.drop(list(df.filter(regex='Rank')))] \
        .drop(columns=['Conference', 'Team'])
    df = df[df.columns.drop(list(df.filter(regex='-Raw')))]
    df['abbreviation'] = df['abbreviation'].map(get_team_from_abbr)
    df['abbreviation'] = df['abbreviation'].str.replace('-ST.', '-STATE')
    df['abbreviation'] = df['abbreviation'].str.replace('STATETE', 'STATE')
    df['abbreviation'] = df['abbreviation'].str.replace('.', '')
    df['abbreviation'] = df['abbreviation'].str.replace('&', '')
    df['abbreviation'] = df['abbreviation'].str.replace("'", '')
    df['abbreviation'] = df['abbreviation'].str.replace('UC-', 'CALIFORNIA-')
    df['abbreviation'] = df['abbreviation'].str.replace('UNC-', 'NORTH-CAROLINA-')
    df['abbreviation'] = df['abbreviation'].str.replace('UT-', 'TEXAS-')
    return df

kenpom_height_df = kp.get_height(browser, season=year)
kenpom_four_df = kp.get_fourfactors(browser, season=year)
kenpom_point_df = kp.get_pointdist(browser, season=year)
kenpom_eff_df = kp.get_efficiency(browser, season=year)
kenpom_height_df = clean_ken(kenpom_height_df)
kenpom_four_df = clean_ken(kenpom_four_df)
kenpom_point_df = clean_ken(kenpom_point_df)
kenpom_eff_df = clean_ken(kenpom_eff_df)

for x in four_convert_to_int:
    kenpom_four_df[x] = pd.to_numeric(kenpom_four_df[x])

for x in height_convert_to_int:
    kenpom_height_df[x] = pd.to_numeric(kenpom_height_df[x])

for x in point_convert_to_int:
    kenpom_point_df[x] = pd.to_numeric(kenpom_point_df[x])

for x in eff_convert_to_int:
    kenpom_eff_df[x] = pd.to_numeric(kenpom_eff_df[x])

for team in Teams(year):
    try:
        df = team.dataframe
        for x in columns_divide_by_minutes:
            df[x] = df[x] / df['minutes_played']
        df = df.drop(columns=['conference', 'name', 'opp_offensive_rating', 'net_rating',
                              'conference_wins', 'conference_losses', 'games_played', 'minutes_played'])
        df = df.merge(kenpom_four_df, on='abbreviation', how='inner')
        df = df.merge(kenpom_height_df, on='abbreviation', how='inner')
        df = df.merge(kenpom_eff_df, on='abbreviation', how='inner')
        df = df.merge(kenpom_point_df, on='abbreviation', how='inner')
        if not df.empty:
            for x in columns_divide_by_hundo:
                df[x] = df[x] / 100
            team_df.append(df)
        else:
            print(team)
    except Exception as e:
        print("FAILED" + str(team))
print(len(team_df))
df = pd.concat(team_df)
df.to_csv('teams{}.csv'.format(year))
