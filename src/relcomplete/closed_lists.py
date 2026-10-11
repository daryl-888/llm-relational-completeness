"""Closed categories with fixed, checkable membership for the parametric (memory-only) setting.

Questions filter these lists by properties computed from the strings or numbers
themselves (first letter, length, atomic-number range), so the gold answer is
derived by code rather than written by hand.
"""
from __future__ import annotations

import periodictable

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
    "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
]

# (number, symbol, name) for all 118 elements, US spellings.
ELEMENTS = [(e.number, e.symbol, e.name.capitalize()) for e in periodictable.elements if e.number > 0]
ELEMENT_ALIASES = {"Aluminium": "Aluminum", "Sulphur": "Sulfur", "Caesium": "Cesium"}

ELEMENT_GROUPS = {
    "alkali metals": [3, 11, 19, 37, 55, 87],
    "alkaline earth metals": [4, 12, 20, 38, 56, 88],
    "halogens": [9, 17, 35, 53, 85, 117],
    "noble gases": [2, 10, 18, 36, 54, 86, 118],
    "lanthanides": list(range(57, 72)),
    "actinides": list(range(89, 104)),
}

# Presidents of the United States in order; Cleveland and Trump served two non-consecutive terms.
PRESIDENTS = [
    "George Washington", "John Adams", "Thomas Jefferson", "James Madison", "James Monroe",
    "John Quincy Adams", "Andrew Jackson", "Martin Van Buren", "William Henry Harrison", "John Tyler",
    "James K. Polk", "Zachary Taylor", "Millard Fillmore", "Franklin Pierce", "James Buchanan",
    "Abraham Lincoln", "Andrew Johnson", "Ulysses S. Grant", "Rutherford B. Hayes", "James A. Garfield",
    "Chester A. Arthur", "Grover Cleveland", "Benjamin Harrison", "Grover Cleveland", "William McKinley",
    "Theodore Roosevelt", "William Howard Taft", "Woodrow Wilson", "Warren G. Harding", "Calvin Coolidge",
    "Herbert Hoover", "Franklin D. Roosevelt", "Harry S. Truman", "Dwight D. Eisenhower", "John F. Kennedy",
    "Lyndon B. Johnson", "Richard Nixon", "Gerald Ford", "Jimmy Carter", "Ronald Reagan",
    "George H. W. Bush", "Bill Clinton", "George W. Bush", "Barack Obama", "Donald Trump", "Joe Biden",
    "Donald Trump",
]


def surname(full: str) -> str:
    return "Van Buren" if full.endswith("Van Buren") else full.split()[-1]


PRESIDENT_SURNAMES = sorted({surname(p) for p in PRESIDENTS})

LISTS = {
    "us_states": US_STATES,
    "elements": [name for _, _, name in ELEMENTS],
    "president_surnames": PRESIDENT_SURNAMES,
}
