from kenpompy.utils import login
import kenpompy.summary as kp
import pandas as pd

# Returns a pandas dataframe containing the efficiency and tempo stats for the current season (https://kenpom.com/summary.php).

# Returns an authenticated browser that can then be used to scrape pages that require authorization.
browser = login('jacksonw765@gmail.com', '9797Jdw!')
fact = kp.get_fourfactors(browser)
master_df = pd.read_csv('team_list_scores2022.csv')
df = df[df.columns.drop(list(df.filter(regex='Test')))]
print(master_df)