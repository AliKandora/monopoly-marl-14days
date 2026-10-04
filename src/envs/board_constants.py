"""Official Monopoly Board Topology, Color Groups, Pricing and Rent Matrices."""

from __future__ import annotations
from typing import Dict, List, Tuple

NUM_TILES = 40

# Tile categories
TILE_TYPE_PROPERTY = 0
TILE_TYPE_RAILROAD = 1
TILE_TYPE_UTILITY = 2
TILE_TYPE_SPECIAL = 3  # GO, Jail, Parking, Go to Jail, Tax, Chance, Chest

# Color Group IDs
GROUP_NONE = -1
GROUP_BROWN = 0
GROUP_LIGHT_BLUE = 1
GROUP_PINK = 2
GROUP_ORANGE = 3
GROUP_RED = 4
GROUP_YELLOW = 5
GROUP_GREEN = 6
GROUP_DARK_BLUE = 7
GROUP_RAILROAD = 8
GROUP_UTILITY = 9

COLOR_GROUPS: Dict[int, List[int]] = {
    GROUP_BROWN: [1, 3],
    GROUP_LIGHT_BLUE: [6, 8, 9],
    GROUP_PINK: [11, 13, 14],
    GROUP_ORANGE: [16, 18, 19],
    GROUP_RED: [21, 23, 24],
    GROUP_YELLOW: [26, 27, 29],
    GROUP_GREEN: [31, 32, 34],
    GROUP_DARK_BLUE: [37, 39],
    GROUP_RAILROAD: [5, 15, 25, 35],
    GROUP_UTILITY: [12, 28],
}

# Tile Names (Classic German / Standard Edition mapped to standard 40 indices)
TILE_NAMES = [
    "LOS (GO)",                   # 0
    "Badstraße",                  # 1 - Brown
    "Gemeinschaftsfeld 1",        # 2
    "Turmstraße",                 # 3 - Brown
    "Einkommensteuer",            # 4
    "Südbahnhof",                 # 5 - Railroad
    "Chausseestraße",             # 6 - Light Blue
    "Ereignisfeld 1",             # 7
    "Elisenstraße",               # 8 - Light Blue
    "Poststraße",                 # 9 - Light Blue
    "Gefängnis / Nur zu Besuch",  # 10
    "Seestraße",                  # 11 - Pink
    "Elektrizitätswerk",          # 12 - Utility
    "Hafenstraße",                # 13 - Pink
    "Neue Straße",                # 14 - Pink
    "Westbahnhof",                # 15 - Railroad
    "Münchner Straße",            # 16 - Orange
    "Gemeinschaftsfeld 2",        # 17
    "Wiener Straße",              # 18 - Orange
    "Berliner Straße",            # 19 - Orange
    "Frei Parken",                # 20
    "Theaterstraße",              # 21 - Red
    "Ereignisfeld 2",             # 22
    "Museumstraße",               # 23 - Red
    "Opernplatz",                 # 24 - Red
    "Nordbahnhof",                # 25 - Railroad
    "Lessingstraße",              # 26 - Yellow
    "Schillerstraße",             # 27 - Yellow
    "Wasserwerk",                 # 28 - Utility
    "Goethestraße",               # 29 - Yellow
    "Gehe ins Gefängnis",         # 30
    "Rathausplatz",               # 31 - Green
    "Hauptstraße",                # 32 - Green
    "Gemeinschaftsfeld 3",        # 33
    "Bahnhofstraße",              # 34 - Green
    "Hauptbahnhof",               # 35 - Railroad
    "Ereignisfeld 3",             # 36
    "Parkstraße",                 # 37 - Dark Blue
    "Zusatzsteuer",               # 38
    "Schlossallee",               # 39 - Dark Blue
]

# Purchase Prices
TILE_PRICES = [
    0,    60,   0,   60,   0,  200,  100,    0,  100,  120,   # 0-9
    0,   140, 150,  140, 160,  200,  180,    0,  180,  200,   # 10-19
    0,   220,   0,  220, 240,  200,  260,  260,  150,  280,   # 20-29
    0,   300, 300,    0, 320,  200,    0,  350,    0,  400    # 30-39
]

# House Costs per Property (0 if not improvable)
HOUSE_COSTS = [
    0,    50,   0,   50,   0,    0,   50,    0,   50,   50,   # 0-9
    0,   100,   0,  100, 100,    0,  100,    0,  100,  100,   # 10-19
    0,   150,   0,  150, 150,    0,  150,  150,    0,  150,   # 20-29
    0,   200, 200,    0, 200,    0,    0,  200,    0,  200    # 30-39
]

# Base Rents: [Rent, 1H, 2H, 3H, 4H, Hotel]
# For Railroads: 25 * 2^(count - 1)
# For Utilities: 4x dice (1 owned) or 10x dice (2 owned)
RENT_TABLE: Dict[int, Tuple[int, int, int, int, int, int]] = {
    1:  (2,   10,  30,   90,  160,  250),
    3:  (4,   20,  60,  180,  320,  450),
    6:  (6,   30,  90,  270,  400,  550),
    8:  (6,   30,  90,  270,  400,  550),
    9:  (8,   40, 100,  300,  450,  600),
    11: (10,  50, 150,  450,  625,  750),
    13: (10,  50, 150,  450,  625,  750),
    14: (12,  60, 180,  500,  700,  900),
    16: (14,  70, 200,  550,  750,  950),
    18: (14,  70, 200,  550,  750,  950),
    19: (16,  80, 220,  600,  800, 1000),
    21: (18,  90, 250,  700,  875, 1050),
    23: (18,  90, 250,  700,  875, 1050),
    24: (20, 100, 300,  750,  925, 1100),
    26: (22, 110, 330,  800,  975, 1150),
    27: (22, 110, 330,  800,  975, 1150),
    29: (24, 120, 360,  850, 1025, 1200),
    31: (26, 130, 390,  900, 1100, 1275),
    32: (26, 130, 390,  900, 1100, 1275),
    34: (28, 150, 450, 1000, 1200, 1400),
    37: (35, 175, 500, 1100, 1300, 1500),
    39: (50, 200, 600, 1400, 1700, 2000),
}

# Group lookup per tile
TILE_TO_GROUP: Dict[int, int] = {}
for g_id, tiles in COLOR_GROUPS.items():
    for t in tiles:
        TILE_TO_GROUP[t] = g_id
