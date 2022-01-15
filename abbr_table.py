abbr_to_team = {'TCU': "TEXAS-CHRISTIAN", 'USC': 'SOUTHERN-CALIFORNIA', "MIAMI": "MIAMI-FL", 'FSU': "FLORIDA-STATE",
                'UCONN': "CONNECTICUT", "UCF": "CENTRAL-FLORIDA", "USF": 'SOUTHERN-FLORIDA', "LSU": "LOUISIANA-STATE",
                'BYU': "BRIGHAM-YOUNG", "UAB": "ALABAMA-BIRMINGHAM", 'OSU': 'OHIO-STATE', 'SMU': 'SOUTHERN-METHODIST',
                'VCU': "VIRGINIA-COMMONWEALTH", "FIU": "FLORIDA-INTERNATIONAL", 'UTEP': 'TEXAS-EL-PASO',
                "MSU": 'MICHIGAN-STATE', }


def get_team_from_abbr(abbr):
    return abbr_to_team.get(abbr, abbr)