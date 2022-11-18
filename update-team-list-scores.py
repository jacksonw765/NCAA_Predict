import time

import pandas as pd
from time import sleep
from sportsipy.ncaab.schedule import Schedule
from tqdm import tqdm

year = 2022
df_teams = pd.read_csv('teams{}.csv'.format(year))
df_teams_columns = df_teams.columns.tolist()
df_teams_columns_new = []
for x in df_teams_columns:
    df_teams_columns_new.append(x + "_2")
all_teams = df_teams['abbreviation'].to_numpy().tolist()
df_sch_list = []
master_df = []


def timer(max):
    for x in tqdm(range(0, max)):
        time.sleep(1)


def get_team(team):
    retval = []
    print(team)
    main_team = df_teams[df_teams['abbreviation'] == team]
    fixed = str(team.strip()).replace(' ', '-').replace('&', '').replace('.', '').replace("'", "").lower()
    sch = Schedule(fixed, year=str(year))
    df_sch = sch.dataframe.dropna(subset=['boxscore_index'])
    df_sch = df_sch[~df_sch.opponent_abbr.str.contains("Non-DI School")]
    df_sch = df_sch[~df_sch.opponent_conference.str.contains("Non-DI School")]
    team_opps = df_sch['opponent_abbr'].to_numpy().tolist()
    for opponent in team_opps:
        team_name = opponent.upper()
        if team_name in all_teams:
            df_sch_select = df_sch[df_sch.opponent_abbr == opponent]
            location = df_sch_select['location'].values.tolist()[0]
            location_home, location_away, location_neutral = 0, 0, 0
            if location == "Away":
                location_away = 1
            elif location == "Home":
                location_home = 1
            elif location == "Neutral":
                location_neutral = 1
            result_arr = [df_sch_select['points_for'].values.tolist()[0],
                          df_sch_select['points_against'].values.tolist()[0],
                          location_home, location_away, location_neutral]
            opp_df = df_teams[df_teams.abbreviation == team_name]
            opp_df.columns = df_teams_columns_new
            result = pd.DataFrame([result_arr],
                                  columns=['points_for', 'points_against', 'location_home', 'location_away',
                                           'location_neutral'])
            append_team = main_team.reset_index(drop=True).merge(opp_df.reset_index(drop=True),
                                                                 left_index=True, right_index=True)
            append_team = pd.concat([append_team, result], axis=1, join='inner')
            retval.append(append_team)
    return retval


for team in all_teams:
    try:
        df = get_team(team)
        master_df.extend(df)
        sleep(2)
    except Exception as e:
        if '429' in str(e):
            val = input('Rate limited! Time to switch VPN...')
            df = get_team(team)
            master_df.append(df)
        else:
            print("error: " + str(e))

master_df = pd.concat(master_df)
master_df.to_csv('team_list_scores{}.csv'.format(year))