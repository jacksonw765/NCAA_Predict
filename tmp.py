from kenpompy.utils import login
import kenpompy.summary as kp
import kenpompy.FanMatch as kpt
import pandas as pd

browser = login('jacksonw765@gmail.com', 'TtEbbYxiA5')
fact = kp.get_efficiency(browser, season=2022)
print(fact)
#master_df = pd.read_csv('team_list_scores2022.csv')
#df = df[df.columns.drop(list(df.filter(regex='Test')))]