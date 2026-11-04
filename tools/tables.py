"""Where the analysis tables live: <data>/tables/<SET>/, numbered in the order they are made.

1_decks.json.gz    tools/deck_table.py     every deck once: format, colours, player, record, predicted wins, card counts
2_cards.csv        tools/card_table.py     every card's features
3_counts.csv.gz    tools/deck_counts.py    every deck's counts (1 x 2), base colours and splash
4_baselines.csv    tools/baselines.py      per colour combination and group: mean, min, percentiles, max of every count
5_compare/         tools/analyse.py        one file per criterion: every field, these decks against the rest
6_findings.csv     tools/analyse.py        what survives the noise checks, across criteria
"""

from companion import DATA_DIR

DECKS, CARDS, COUNTS, BASELINES, COMPARE, FINDINGS = (
    "1_decks.json.gz", "2_cards.csv", "3_counts.csv.gz", "4_baselines.csv", "5_compare", "6_findings.csv")


def path(set_code, name):
    folder = DATA_DIR / "tables" / set_code
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name
