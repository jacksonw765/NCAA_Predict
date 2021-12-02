import pandas as pd
from sportsipy.ncaab.schedule import Schedule

df_teams = pd.read_csv('teams.csv')
df_teams_columns = df_teams.columns.tolist()
df_teams_columns_new = []
for x in df_teams_columns:
    df_teams_columns_new.append(x + "_2")
df_teams = df_teams.dropna()
all_teams = df_teams['abbreviation'].to_numpy().tolist()
all_teams_tmp = all_teams[:10]
df_sch_list = []
master_df = []
for team in all_teams:
    print(team)
    main_team = df_teams[df_teams['abbreviation'] == team]
    fixed = str(team.strip()).replace(' ', '-').replace('&', '').replace('.', '').replace("'", "").lower()
    try:
        sch = Schedule(fixed, year='2022')
        df_sch = sch.dataframe.dropna(subset=['boxscore_index'])
        df_sch = df_sch[~df_sch.opponent_abbr.str.contains("Non-DI School")]
        team_opps = df_sch['opponent_abbr'].to_numpy().tolist()
        for opponent in team_opps:
            team_name = opponent.upper()
            if team_name in all_teams:
                df_sch_select = df_sch[df_sch.opponent_abbr == opponent]
                result_arr = [df_sch_select['points_for'].values.tolist()[0],
                              df_sch_select['points_against'].values.tolist()[0]]
                opp_df = df_teams[df_teams.abbreviation == team_name]
                opp_df.columns = df_teams_columns_new
                result = pd.DataFrame([result_arr], columns=['points_for', 'points_against'])
                append_team = main_team.reset_index(drop=True).merge(opp_df.reset_index(drop=True),
                                                                     left_index=True, right_index=True)
                append_team = pd.concat([append_team, result], axis=1, join='inner')
                master_df.append(append_team)
    except Exception as e:
        print(fixed)

master_df = pd.concat(master_df)
master_df.to_csv('team_list_scores2.csv')