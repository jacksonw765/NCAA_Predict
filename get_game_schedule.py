from bs4 import BeautifulSoup
import requests
import pandas as pd


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
            home_teams.append(_clean_str(spans[0].text))
            away_teams.append(_clean_str(spans[-1].text))
    away_teams.pop(0)
    home_teams.pop(0)

    return away_teams, home_teams

def _clean_str(span):
    return str(span.strip()).replace(' ', '-').replace('&', '').replace('.', '').replace("'", "").upper()

