"""Define all constants."""

PRED_KEYS = {
    "neutrophil": 1,
    "epithelial-cell": 2,
    "lymphocyte": 3,
    "plasma-cell": 4,
    "eosinophil": 5,
    "connective-tissue-cell": 6,
    "mitosis": 7,
    "empty": 8,
    "filtered_lymphocyte": 9,
}
LOOKUP = ["bg", *list(PRED_KEYS.keys())]
LUP_CA = [0] * 256

CA_DICT = {
    "Adipose": 0,
    "Background": 1,
    "Debris": 2,
    "Lymphocytes": 3,
    "Mucin": 4,
    "Muscle": 5,
    "Normal mucosa": 6,
    "Stroma": 7,
    "Tumor": 8,
}

for i, j in zip([26, 78, 86, 126, 127, 129, 153, 190, 255], [8, 3, 6, 7, 4, 0, 1, 5, 2]):
    LUP_CA[i] = j

RADII = [20, 50, 100, 200, 500]  # micrometers
