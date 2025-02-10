# PhonoGlyphe

PhonoGlyphe is a G2P (grapheme to phoneme) 6M transformer model meant as a fallback method for the [Misaki](https://github.com/hexgrad/misaki/) G2P engine. It currently only supports the English language.

![](/img/phonoglyphe.png)

## Usage

```python
from misaki import en
from PhonoGlyphe import G2PModel

fallback = G2PModel(device="cpu")	# Note: with it's small size, PhonoGlyphe is often faster on CPU
g2p = en.G2P(trf=False, british=False, fallback=fallback)

print(f"Phonemes: {phonemes}")
```

You can easily listen to the final result using `[.](/<GENERATED PHONEMES>/)` in this [HF Space](https://huggingface.co/spaces/hexgrad/Kokoro-TTS/).

## Training

Phonoglyphe was trained using the English dictionnaries of the [Misaki](https://github.com/hexgrad/misaki/) project.
The training code with be released soon™.