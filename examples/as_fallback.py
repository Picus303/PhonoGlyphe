"""
pip install misaki[en] phonoglyphe
"""
from misaki import en
from phonoglyphe import G2PModel

fallback = G2PModel(device="cpu")	# Note: with its small size, PhonoGlyphe is often faster on CPU
g2p = en.G2P(trf=False, british=False, fallback=fallback)

text = "Misaki is a G2P engine designed for Kokoro models."
phonemes, tokens = g2p(text)

print(f"Phonemes: {phonemes}")