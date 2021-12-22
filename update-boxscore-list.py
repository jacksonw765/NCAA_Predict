import pandas as pd
from sportsipy.ncaab.boxscore import Boxscore
from sportsipy.ncaab.schedule import Schedule
from sportsipy.ncaab.teams import Teams

master_df = []
year = 2018
for team in Teams(year):
    try:
        team_df = team.dataframe
        team_abbr = team_df['abbreviation'].values.tolist()[0]
        fixed = str(team_abbr.strip()).replace(' ', '-').replace('&', '').replace('.', '').replace("'", "").lower()
        print(fixed)
        sch = Schedule(fixed, year=str(year))
        df_sch = sch.dataframe.dropna(subset=['boxscore_index'])
        df_sch = df_sch[~df_sch.opponent_abbr.str.contains("Non-DI School")]
        df_sch = df_sch[~df_sch.opponent_conference.str.contains("Non-DI School")]
        boxscores = df_sch['boxscore_index'].to_numpy().tolist()
        team_opps = df_sch['opponent_abbr'].to_numpy().tolist()
        for i, boxscore in enumerate(boxscores):
            opp_team = team_opps[i].upper()
            score_df = Boxscore(boxscore).dataframe
            location = df_sch['location'].values.tolist()[i]
            # prob a horrible way to do this but idc its late
            team_1_df, team_2_df = None, None
            location_home, location_away, location_neutral = 0, 0, 0
            if location == "Away":
                team_1_df = score_df.filter(regex='away')
                team_1_df.columns = team_1_df.columns.str.replace("away", "team1")
                team_2_df = score_df.filter(regex='home')
                team_2_df.columns = team_2_df.columns.str.replace("home", "team2")
                location_away = 1
            elif location == "Home":
                team_1_df = score_df.filter(regex='home')
                team_1_df.columns = team_1_df.columns.str.replace("home", "team1")
                team_2_df = score_df.filter(regex='away')
                team_2_df.columns = team_2_df.columns.str.replace("away", "team2")
                location_home = 1
            elif location == "Neutral":
                location_neutral = 1
                # really weird way to do this ik should be ashamed to do it this way
                if team_abbr.lower() in boxscore:
                    team_1_df = score_df.filter(regex='home')
                    team_1_df.columns = team_1_df.columns.str.replace("home", "team1")
                    team_2_df = score_df.filter(regex='away')
                    team_2_df.columns = team_2_df.columns.str.replace("away", "team2")
                else:
                    team_1_df = score_df.filter(regex='away')
                    team_1_df.columns = team_1_df.columns.str.replace("away", "team1")
                    team_2_df = score_df.filter(regex='home')
                    team_2_df.columns = team_2_df.columns.str.replace("home", "team2")

            score_df = team_1_df.reset_index(drop=True).merge(team_2_df.reset_index(drop=True), left_index=True,
                                                              right_index=True)
            result_arr = [team_abbr, df_sch['points_for'].values.tolist()[i], opp_team,
                          df_sch['points_against'].values.tolist()[i], location_home, location_away, location_neutral]
            result = pd.DataFrame([result_arr], columns=['team1', 'points_for', 'team2', 'points_against',
                                                         'location_home', 'location_away', 'location_neutral'])
            append_df = score_df.reset_index(drop=True).merge(result.reset_index(drop=True),
                                                              left_index=True, right_index=True)
            master_df.append(append_df)
    except Exception as e:
        print(e)

df = pd.concat(master_df)
df = df.drop(columns=['team1_ranking', 'team2_ranking', 'team1_wins', 'team2_wins', 'team1_points', 'team2_points'])
df.to_csv('teams_list_boxscores{}.csv'.format(year))