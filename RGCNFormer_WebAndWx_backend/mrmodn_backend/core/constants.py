"""
中文：RNA 分类常量定义。包含核苷酸分组、修饰名称映射、标签索引等共享常量。
English: RNA classification constants. Contains nucleotide groups, modification name mappings, label indices, and other shared constants.
"""


# 12-class modification name mapping (model index -> modification name)
# Consistent with human_make_npy.py MOD_TO_INDEX (mod_index - 1)
MOD_NAMES = {
    0: 'Am',     1: 'Atol',   2: 'Cm',
    3: 'Gm',     4: 'Tm',     5: 'Y',
    6: 'ac4C',   7: 'm1A',    8: 'm5C',
    9: 'm6A',    10: 'm6Am',  11: 'm7G'
}

# 4-Class Group Names (Nucleotide Groups)
NUCLEOTIDE_GROUP_NAMES = ['A', 'C', 'G', 'U']

# Model Index (0-11) to Nucleotide Group Name
INDEX_TO_NUCLEOTIDE = {
    0: 'A', 1: 'A',
    2: 'C',
    3: 'G',
    4: 'U', 5: 'U',
    6: 'C',
    7: 'A',
    8: 'C',
    9: 'A', 10: 'A',
    11: 'G'
}

# Group Name to original IDs (for reference, 1-indexed from dataset)
NUCLEOTIDE_GROUPS = {
    'A': [1, 2, 8, 10, 11],
    'C': [3, 7, 9],
    'G': [4, 12],
    'U': [5, 6]
}

# Group to 4-Class Index Mapping (Order for y_4class)
GROUP_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

# Index to Group Mapping (reverse of above)
INDEX_TO_GROUP = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

# Number of classes per group (for hierarchical derivation)
GROUP_SIZES = {
    'A': 5,
    'C': 3,
    'G': 2,
    'U': 2
}

# Mapping from group to the indices (0-11) of classes in that group
GROUP_TO_CLASS_INDICES = {
    'A': [0, 1, 7, 9, 10],
    'C': [2, 6, 8],
    'G': [3, 11],
    'U': [4, 5]
}

# Plant dataset only has 3 valid classes (Y, m5C, m6A)
PLANT_VALID_CLASS_INDICES = [5, 8, 9]

# Label mapping: original label ID (1-12) -> model index (0-11)
# Values in 1001loc.npy are 1-12 (mod_index); LABEL_MAPPING converts to 0-11
LABEL_MAPPING = {
    1: 0,   # Am
    2: 1,   # Atol
    3: 2,   # Cm
    4: 3,   # Gm
    5: 4,   # Tm
    6: 5,   # Y
    7: 6,   # ac4C
    8: 7,   # m1A
    9: 8,   # m5C
    10: 9,  # m6A
    11: 10, # m6Am
    12: 11  # m7G
}
