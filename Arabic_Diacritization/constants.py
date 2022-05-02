''' This file contains all the constants that will be used by all models '''

ARABIC_CHAR = "ىعظحرسيشضق ثلصطكآماإهزءأفؤغجئدةخوبذتن"

NUMBERS = "0123456789٠١٢٣٤٥٦٧٨٩"

PUNCTUATIONS = [
    ".",    "،",    ":",    "؛",
    "-",    "–",    "«",    "»",
    "~",    "؟",    "!",    "*", 
    "(",    ")",    "[",    "]", 
    "{",    "}",    ";",    "\n",
    "'",    "\"",    "`",   " " ,
    "/",    ",",
]

# Diacritics
FATHATAN = u'\u064b'
DAMMATAN = u'\u064c'
KASRATAN = u'\u064d'
FATHA = u'\u064e'
DAMMA = u'\u064f'
KASRA = u'\u0650'
SHADDA = u'\u0651'
SUKUN = u'\u0652'

HARAQAT = [FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA, KASRA, SHADDA, SUKUN]

ALLCHARS = HARAQAT + list(ARABIC_CHAR) + PUNCTUATIONS + list(NUMBERS)
VOCAB_SIZE = len(ALLCHARS) + 2 # including <PAD> and <UKN>

# 15 possible Diacritics
DIACRITICS_CLASS = {
    ""              : 0  , # No Diacritic
    FATHA           : 1  , # Fatha
    FATHATAN        : 2  , # Fathatah
    DAMMA           : 3  , # Damma
    DAMMATAN        : 4  , # Dammatan
    KASRA           : 5  , # Kasra
    KASRATAN        : 6  , # Kasratan
    SUKUN           : 7  , # Sukun
    SHADDA          : 8  , # Shadda
    SHADDA+FATHA    : 9  , # Shadda + Fatha
    SHADDA+FATHATAN : 10 , # Shadda + Fathatah
    SHADDA+DAMMA    : 11 , # Shadda + Damma
    SHADDA+DAMMATAN : 12 , # Shadda + Dammatan
    SHADDA+KASRA    : 13 , # Shadda + Kasra
    SHADDA+KASRATAN : 14 , # Shadda + Kasratan
}

BASE_PATH = "./Arabic_Diacritization"
TRAIN_DIR = f"{BASE_PATH}/.Tashkeel/train.txt"
VAL_DIR = f"{BASE_PATH}/.Tashkeel/valid.txt"
TEST_DIR = f"{BASE_PATH}/.Tashkeel/test.txt"
PICKLE_LOCATION = f"{BASE_PATH}/.Tashkeel"

def CHAR_IDX():
    char2idx = {}
    idx2char = {}

    for i, char in enumerate(ALLCHARS):
        char2idx[char] = i + 1
        idx2char[i + 1] = char
    
    return char2idx, idx2char