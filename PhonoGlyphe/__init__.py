import os
import json
import torch
from typing import Tuple, Any

from .architecture import Transformer
from .utils import build_transformer

LIBRARY_PATH = os.path.dirname(__file__)

CONFIG_PATH = LIBRARY_PATH + "/config.json"
TOKENIZERS_PATH = LIBRARY_PATH +"/tokenizers.json"
WEIGHTS_PATH = LIBRARY_PATH + "/PhonoGlyphe_V1_i40.pth"

PREDICTION_RATING = 2

ERROR_RESPONSE = "❓"

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
TARGET_TOKEN = "<TARGET>"

SPECIAL_TOKENS = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, TARGET_TOKEN]



class G2PModel:
	def __init__(self, model_path: str = None, device: str = "cpu"):
		"""
		Initializes the G2PModel object.
		This model is used to predict the phonemes of a given word.
		It's meant to be used as a fallback method for the Misaki G2P engine.

		Args:
			model_path (str): The path to the model weights.
			device (str): The device to use for the model.
		"""
		# Select the device
		self.device = device if torch.cuda.is_available() else "cpu"
		if self.device != device:
			print(f"Warning: {device} is not available. Using {self.device} instead.")

		# Load the configuration
		with open(CONFIG_PATH, "r") as config_file:
			self.config: dict = json.load(config_file)

		# Load the tokenizers
		with open(TOKENIZERS_PATH, "r") as tokenizers_file:
			self.tokenizers: dict = json.load(tokenizers_file)

		# Build the model
		self.model: Transformer = build_transformer(**self.config)
		if not os.path.exists(WEIGHTS_PATH):
			raise FileNotFoundError("The model weights could not be found.")

		# Load the weights
		model_path = model_path if model_path is not None else WEIGHTS_PATH
		self.model.load_state_dict(torch.load(model_path, weights_only=True))
		self.model.to(self.device)
		self.model.eval()

		# prepare attention masks
		self.masks = torch.ones((self.context_length, 1, self.context_length, self.context_length), device=device)
		for i in range(self.context_length - 1):
			self.masks[i, :, :, i+1:] = 0


	@property
	def context_length(self) -> int:
		return self.config["context_length"]

	@property
	def alphabet_tokenizer(self) -> int:
		return self.tokenizers["alphabet_t2i"]

	@property
	def phoneme_tokenizer_in(self) -> int:
		return self.tokenizers["phoneme_t2i"]

	@property
	def phoneme_tokenizer_out(self) -> int:
		return self.tokenizers["phoneme_i2t"]

	@property
	def supported_characters(self) -> str:
		return self.tokenizers["supported_characters"]


	def predict(self, text: str) -> Tuple[str, int]:
		"""
		Inference pipeline for the G2P model.

		Args:
			text (str): The input text to predict.

		Returns:
			A tuple containing the predicted phonemes and the prediction rating.
		"""

		try:
			with torch.no_grad():
				# Check the input text
				assert len(text) < self.context_length, f"Input text is too long ({len(text)} > {self.context_length})"
				assert all(c in self.supported_characters for c in text), f"Input text contains unknown characters"

				padding_token = self.alphabet_tokenizer[PAD_TOKEN]
				target_token = self.phoneme_tokenizer_in[TARGET_TOKEN]

				# Tokenize the input text
				word_tensor = torch.tensor([self.alphabet_tokenizer[c] for c in text], device=self.device).unsqueeze(0)

				padding_length = self.context_length - word_tensor.size(1)
				padding_tensor = torch.full((1, padding_length), padding_token, dtype=torch.long, device=self.device)

				word_tensor = torch.cat([word_tensor, padding_tensor], dim=1)

				# Encode the input text
				mask = self.masks[len(text) - 1]
				encoder_output = self.model.encode(word_tensor, mask)

				# Decode the phonemes
				output = [SOS_TOKEN]

				phoneme_tensor = torch.full((1, self.context_length), padding_token, dtype=torch.long, device=self.device)

				while (output[-1] != EOS_TOKEN) and (len(output) < self.context_length):
					phoneme_tensor[0, len(output) - 1] = self.phoneme_tokenizer_in[output[-1]]
					phoneme_tensor[0, len(output)] = target_token

					mask = self.masks[len(output) - 1]
					decoder_output = self.model.decode(phoneme_tensor, encoder_output, mask)
					model_output = self.model.project(decoder_output)

					_, next_token = model_output[0].max(dim=0)
					output.append(self.phoneme_tokenizer_out[str(next_token.item())])

				output = ''.join(output)

				# Remove special tokens
				for special_token in SPECIAL_TOKENS:
					output = output.replace(special_token, "")

				return output, PREDICTION_RATING

		except AssertionError as e:
			return ERROR_RESPONSE, 0


	def __call__(self, word: Any) -> Tuple[str, int]:
		return self.predict(word.text)