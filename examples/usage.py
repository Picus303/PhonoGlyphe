"""
pip install phonoglyphe
"""

import phonoglyphe

phonemizer = phonoglyphe.G2PModel()
phonemes = phonemizer.predict("Hello world")
print(phonemes)