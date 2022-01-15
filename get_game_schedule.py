from bs4 import BeautifulSoup
import requests
from abbr_table import get_team_from_abbr

def has_numbers(string):
    return any(char.isdigit() for char in string)


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
        for i, tr in enumerate(table_data):
            if i > 0:
                spans = tr.find_all("span")
                cln = []
                for span in spans:
                    if not has_numbers(span.text):
                        team = span.text.replace(' ', '-')
                        if len(team) <= 5:
                            team = get_team_from_abbr(team)
                        cln.append(team)
                home_teams.append(_clean_str(cln[-1]))
                away_teams.append(_clean_str(cln[0]))
    assert len(home_teams) == len(away_teams)
    return away_teams, home_teams

def get_favored_team_for_date(date):
    ESPN_URL = 'https://www.espn.com/mens-college-basketball/scoreboard/_/date/'.format(date)
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = requests.get(ESPN_URL, headers=headers)
    soup = BeautifulSoup(req.text, "lxml")
    team_1 = []
    team_2 = []
    sections = soup.find_all('div', {'class':"scoreboard-wrapper"})
    for sec in sections:
        tables = sec.find_all('table')
        for table in tables:
            trs = table.find_all('tr')
            print(trs)
        #divs = sec.find_all('div')
        #for div in divs:
        #    if 'stat--full-width' in str(div):
        #        print(div)
    return team_1, team_2

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
                        tmp_name = '-'
                        away_abbr.update({tmp[-1]:tmp_name.join(tmp[:-1])})
                if "lpos=mens-college-basketball:schedule:score" in str(a_tag):
                    res = str(a_tag.text).replace(',', '').split(" ")
                    away = ''
                    if '#' in spans[0].text:
                        away = _clean_str(spans[1].text)
                        if len(away) <= 5:
                            away = get_team_from_abbr(away)
                    else:
                        away = _clean_str(spans[0].text)
                        if len(away) <= 5:
                            away = get_team_from_abbr(away)
                    home = _clean_str(spans[-1].text)
                    if len(home) <= 5:
                        home = get_team_from_abbr(home)
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
                        if "Canceled" not in res and 'Postponed' not in res and 'Forfeit' not in res:
                            print("Failed to get: " + str(res) + str(e))
    assert len(team_1) == len(team_2)
    return team_1, team_2, is_neutral_loc


def _clean_str(span):
    return str(span.split('#')[0].strip()).replace(' ', '-').replace('&', '').replace('.', '').replace("'", "").upper()

#teams_1, teams_2 = get_favored_team_for_date(20220112)
#teams_1, teams_2 = get_schedules_for_date(20220112)
#print(teams_1)