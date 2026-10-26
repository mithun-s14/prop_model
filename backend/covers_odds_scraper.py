"""
Live NBA spread/total from covers.com.

The odds page is server-rendered, so a plain GET works. Each game is a
`tr.oddsGameRow` in `#spread-table` / `#total-table`; every sportsbook column
is a `td[data-book]` whose `.home-cell` link text starts with the line
("-3.5", "u 232.5"). We take the median across books as the consensus line.
"""
from statistics import median

import requests
from bs4 import BeautifulSoup

COVERS_NBA_ODDS_URL = 'https://www.covers.com/sport/basketball/nba/odds'

# covers abbreviation -> Basketball Reference abbreviation (what our caches use)
COVERS_TO_BBREF = {
    'GS': 'GSW', 'CHA': 'CHO', 'BK': 'BRK', 'BKN': 'BRK', 'NY': 'NYK',
    'NO': 'NOP', 'SA': 'SAS', 'PHX': 'PHO', 'UTAH': 'UTA', 'WSH': 'WAS',
}


def _norm(abbr):
    abbr = abbr.strip().upper()
    return COVERS_TO_BBREF.get(abbr, abbr)


def _home_lines(table):
    """Yield (away, home, consensus home-cell line) for each game row in a table."""
    for row in table.select('tr.oddsGameRow'):
        teams = [s.get_text(strip=True) for s in row.select('.left-cell strong')]
        if len(teams) != 2:
            continue
        lines = []
        for td in row.find_all('td', attrs={'data-book': True}):
            link = td.select_one('.home-cell a')
            text = link.find(string=True, recursive=False) if link else None
            try:
                lines.append(float(text.strip().lstrip('ou ').replace('PK', '0')))
            except (AttributeError, ValueError):
                continue
        if lines:
            yield _norm(teams[0]), _norm(teams[1]), median(lines)


def parse_game_lines(html):
    """Return {(away, home): {'home_spread': float, 'total': float}} from the odds page HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    games = {}
    for table_id, key in (('spread-table', 'home_spread'), ('total-table', 'total')):
        table = soup.find(id=table_id)
        if table is None:
            continue
        for away, home, line in _home_lines(table):
            # setdefault: the page can list later games too; keep the first (soonest)
            games.setdefault((away, home), {}).setdefault(key, line)
    return games


def fetch_game_lines(timeout=10):
    resp = requests.get(COVERS_NBA_ODDS_URL, timeout=timeout,
                        headers={'User-Agent': 'Mozilla/5.0'})
    resp.raise_for_status()
    return parse_game_lines(resp.text)


def get_team_lines(team, opponent, games=None):
    """
    Spread (from `team`'s perspective, negative = favored) and total for the
    game between `team` and `opponent`. Returns (None, None) for any line not found.
    """
    if games is None:
        try:
            games = fetch_game_lines()
        except requests.RequestException as e:
            print(f"Could not fetch covers.com odds: {e}")
            return None, None
    for (away, home), lines in games.items():
        if {away, home} == {team, opponent}:
            spread = lines.get('home_spread')
            if spread is not None and team == away:
                spread = -spread
            return spread, lines.get('total')
    return None, None


if __name__ == '__main__':
    for (away, home), lines in fetch_game_lines().items():
        print(f"{away} @ {home}: {lines}")
