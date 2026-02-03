PADDING_INDEX = 999

ELEMENTS_HASH = {
    "C": 0,
    "H": 1,
    "O": 2,
    "N": 3,
    "F": 4,
    "S": 5,
    "Cl": 6,
    "Br": 7,
    "P": 8,
    "I": 9,
    "B": 10,
}

# Index to atom type (inverse mapping)
ELEMENTS_HASH_INV = {
    0: 'C',
    1: 'H',
    2: 'O',
    3: 'N',
    4: 'F',
    5: 'S',
    6: 'Cl',
    7: 'Br',
    8: 'P',
    9: 'I',
    10: 'B'
}

# For now, using tha atomic radii from https://github.com/gnina/libmolgrid/blob/master/src/atom_typer.cpp
# which is the same used in AutoDock v4.
radiusSingleAtom = {
    "MOL": {
        "C": 2.0,
        "H": 1.0,
        "O": 1.6,
        "N": 1.75,
        "F": 1.545,
        "S": 2.0,
        "Cl": 2.045,
        "Br": 2.165,
        "P": 2.1,
        "I": 2.36,
        "B": 2.04,
    }
}

# Bond lengths in pm (divide by 100 for Angstrom); from baselines_evaluation.py
BOND_LENGTHS_PM = {
    'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
          'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
          'Cl': 127, 'Br': 141, 'I': 161},
    'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
          'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
          'I': 214},
    'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
          'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
    'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
          'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
          'I': 194},
    'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
          'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
          'I': 187},
    'B': {'H': 119, 'Cl': 175},
    'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
           'F': 160, 'Cl': 202, 'Br': 215, 'I': 243},
    'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
           'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
           'Br': 214},
    'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
          'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
          'I': 234},
    'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
           'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
    'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
          'S': 210, 'F': 156, 'N': 177, 'Br': 222},
    'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
          'S': 234, 'F': 187, 'I': 266},
    'As': {'H': 152}
}

# Default bond length threshold (Angstrom) for unknown atom pairs
DEFAULT_BOND_LENGTH_THRESHOLD = 2.0  # Angstrom
