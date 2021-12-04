import pandas as pd
from sportsipy.ncaab.boxscore import Boxscore
from sportsipy.ncaab.schedule import Schedule

df_teams = pd.read_csv('teams2022.csv')
all_teams = df_teams['abbreviation'].to_numpy().tolist()[:10]
master_df = []
for team in all_teams:
    print(team)
    main_team = df_teams[df_teams['abbreviation'] == team]
    fixed = str(team.strip()).replace(' ', '-').replace('&', '').replace('.', '').replace("'", "").lower()
    try:
        sch = Schedule(fixed, year='2022')
        df_sch = sch.dataframe
        df_sch = df_sch[~df_sch.opponent_abbr.str.contains("Non-DI School")]
        df_sch = df_sch[~df_sch.opponent_conference.str.contains("Non-DI School")]
        boxscores = df_sch['boxscore_index'].to_numpy().tolist()
        team_opps = df_sch['opponent_abbr'].to_numpy().tolist()
        for i, boxscore in enumerate(boxscores):
            opp_team = team_opps[i]
            score = Boxscore(boxscore).dataframe
            master_df.append(score)
        df = pd.concat(master_df)
        df = df.drop(columns=['location', 'losing_abbr', 'losing_name', 'winner', 'winning_abbr', ])
    except Exception as e:
        print(e)