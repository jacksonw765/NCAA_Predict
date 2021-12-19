from bs4 import BeautifulSoup
import requests

def get_schedules_for_date(date):
    # fill {} with year, month, day so 20211124
    ESPN_URL = 'https://www.espn.com/mens-college-basketball/schedule/_/date/{}/group/50'.format(date)
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = requests.get(ESPN_URL, headers=headers)
    soup = BeautifulSoup(req.text, "lxml")

    table_rows = soup.find_all('table')
    away_teams = []
    home_teams = []
    for table in table_rows:
        table_data = table.find_all("tr")
        for tr in table_data:
            spans = tr.find_all("span")
            home_teams.append(_clean_str(spans[-1].text))
            away_teams.append(_clean_str(spans[0].text))
    away_teams.pop(0)
    home_teams.pop(0)

    return away_teams, home_teams


def get_scores_for_date(date):
    # fill {} with year, month, day so 20211124
    ESPN_URL = 'https://www.espn.com/mens-college-basketball/schedule/_/date/{}/group/50/date/'.format(date)
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = requests.get(ESPN_URL, headers=headers)
    soup = BeautifulSoup(req.text, "lxml")
    team_1 = []
    team_2 = []
    is_neutral_loc = []
    table_rows = soup.find_all('table')
    for table in table_rows:
        table_data = table.find_all("tr")
        for tr in table_data:
            spans = tr.find_all("span")
            is_neutral = tr.attrs.get('data-is-neutral-site')
            away_abbr = {}
            for a_tag in tr.find_all('a'):
                if "lpos=mens-college-basketball:schedule:team" in str(a_tag):
                    if a_tag.text != '' and len(away_abbr) != 1:
                        tmp = a_tag.text.split(' ')
                        away_abbr.update({tmp[1]:tmp[0]})
                if "lpos=mens-college-basketball:schedule:score" in str(a_tag):
                    res = str(a_tag.text).replace(',', '').split(" ")
                    if '#' in spans[0].text:
                        away = _clean_str(spans[1].text)
                    else:
                        away = _clean_str(spans[0].text)
                    home = _clean_str(spans[-1].text)
                    try:
                        if list(away_abbr.keys())[0] == res[0]:
                            team_1.append({away: int(res[1])})
                            team_2.append({home: int(res[3])})
                        else:
                            team_1.append({away: int(res[3])})
                            team_2.append({home: int(res[1])})
                        if is_neutral == 'true':
                            is_neutral_loc.append(1)
                        else:
                            is_neutral_loc.append(0)
                    except Exception as e:
                        if "Canceled" not in res and 'Postponed' not in res:
                            print("Failed to get: " + str(res) + str(e))
    assert len(team_1) == len(team_2)
    return team_1, team_2, is_neutral_loc



def _clean_str(span):
    return str(span.split('#')[0].strip()).replace(' ', '-').replace('&', '').replace('.', '').replace("'", "").upper()
