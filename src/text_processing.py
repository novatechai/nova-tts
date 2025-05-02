# src/text_processing.py

import re
import inflect
import phonemizer
from transformers import AutoTokenizer
import torch # Added for attention mask
import sys
sys.path.append('.') # To import config from the root directory run
import src.config as config
import os # Make sure os is imported
from phonemizer.backend.espeak.wrapper import EspeakWrapper # Import the wrapper

class TextProcessor:
    """Handles text normalization, phonemization, and tokenization."""

    def __init__(self, config):
        print("Initializing Text Processor...")
        self.config = config
        self._whitespace_re = re.compile(r"\s+")
        self._inflect_engine = inflect.engine()
        self.tokenizer = self._load_tokenizer()
        self.phonemizer = self._load_phonemizer()
        print("Text Processor Initialized.")

    def _load_tokenizer(self):
        """Loads the Hugging Face tokenizer specified in the config."""
        try:
            print(f"Loading tokenizer: {self.config.TEXT_ENCODER_MODEL}")
            tokenizer_instance = AutoTokenizer.from_pretrained(
                self.config.TEXT_ENCODER_MODEL, trust_remote_code=True
            )
            # Add pad token if missing (common for some models like Qwen)
            if tokenizer_instance.pad_token is None:
                 tokenizer_instance.pad_token = tokenizer_instance.eos_token
                 print(f"Set pad_token to eos_token: {tokenizer_instance.pad_token}")
            print("Tokenizer loaded successfully.")
            return tokenizer_instance
        except Exception as e:
            print(f"ERROR loading tokenizer: {e}")
            # Decide how to handle this - maybe raise exception?
            raise RuntimeError(f"Failed to load tokenizer: {e}")

    def _load_phonemizer(self):
        """Loads the Espeak phonemizer backend."""
        try:
            espeak_lib_path = "/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib"
            if sys.platform == "darwin" and os.path.exists(espeak_lib_path):
                print(f"Attempting to set espeak library path to: {espeak_lib_path}")
                try:
                    EspeakWrapper.set_library(espeak_lib_path)
                    print("Successfully set espeak library path.")
                except Exception as e_setlib:
                    print(f"Warning: Failed to set espeak library path: {e_setlib}")

            print("Initializing phonemizer with espeak backend...")
            phonemizer_instance = phonemizer.backend.EspeakBackend(
                language='en-us', preserve_punctuation=True, with_stress=True, language_switch='remove-flags'
            )
            print("Phonemizer initialized successfully.")
            return phonemizer_instance
        except Exception as e:
            print(f"ERROR initializing phonemizer: {e}")
            print("Please ensure 'espeak' or 'espeak-ng' backend is installed.")
            # Decide how to handle - raise error or allow fallback to chars?
            # raise RuntimeError(f"Failed to initialize phonemizer: {e}")
            print("WARNING: Phonemizer failed to load. Phonemization will be skipped.")
            return None # Allow fallback to character encoding

    def _expand_numbers(self, text):
        """Converts numbers to words using the inflect library."""
        words = []
        for word in text.split():
            if word.isdigit():
                try:
                    num_word = self._inflect_engine.number_to_words(word)
                    words.append(num_word)
                except Exception:
                    words.append(word) # Keep original if conversion fails
            else:
                words.append(word)
        return ' '.join(words)

    def _normalize_text(self, text):
        """Basic text normalization: lowercase, expand numbers, collapse whitespace."""
        # Filter out tags like <giggles>, <laughs>, etc.
        text = re.sub(r'<[^>]+>', ' ', text) # Replace tags with a space
        
        text = text.lower()
        text = self._expand_numbers(text)
        text = re.sub(self._whitespace_re, ' ', text)
        text = text.strip()
        return text

    def _text_to_phonemes(self, text):
        """Converts normalized text to phonemes."""
        if self.phonemizer is None:
            print("WARNING: Phonemizer not available. Returning normalized text instead of phonemes.")
            return text
        try:
            # --- DEBUG PRINT REMOVED ---
            # print(f"DEBUG PHONEMIZER INPUT: '{text}'") 
            # --- END DEBUG PRINT ---
            phonemes = self.phonemizer.phonemize([text], strip=True)[0]
            # --- DEBUG PRINT REMOVED ---
            # print(f"DEBUG PHONEMIZER OUTPUT: '{phonemes}'")
            # --- END DEBUG PRINT ---
            return phonemes
        except Exception as e:
            print(f"ERROR during phonemization for text '{text[:50]}...': {e}")
            return text # Fallback to normalized text on error

    def _tokenize(self, sequence):
        """Tokenizes a sequence (text or phonemes)."""
        if self.tokenizer is None:
            # This shouldn't happen due to check in __init__, but defensive check
            raise RuntimeError("Tokenizer not initialized.") 
        try:
            # Let's assume max_length can be handled by padding/truncation in collate_fn later
            encoding = self.tokenizer(sequence, return_tensors=None, add_special_tokens=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            return input_ids, attention_mask
        except Exception as e:
            print(f"ERROR during tokenization for sequence '{sequence[:50]}...': {e}")
            # Return empty lists or handle differently?
            return [], []

    def process(self, raw_text, phonemes=True):
        """Applies the full text processing pipeline.
        
        Args:
            raw_text (str): The input text.
            phonemes (bool): Whether to convert to phonemes before tokenizing.
                             Defaults to True. If False or if phonemizer failed,
                             tokenizes normalized text directly.
                             
        Returns:
            tuple: (input_ids (list[int]), attention_mask (list[int]))
        """
        normalized = self._normalize_text(raw_text)
        
        sequence_to_tokenize = normalized
        if phonemes and self.phonemizer:
            phoneme_sequence = self._text_to_phonemes(normalized)
            # Only use phonemes if phonemization didn't fallback to text
            if phoneme_sequence != normalized: 
                sequence_to_tokenize = phoneme_sequence
            else:
                print("Phonemization resulted in fallback to text, tokenizing normalized text.")
        elif phonemes and not self.phonemizer:
             print("Phonemizer unavailable, tokenizing normalized text.")
        
        input_ids, attention_mask = self._tokenize(sequence_to_tokenize)
        
        return input_ids, attention_mask

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("\n--- Testing TextProcessor Class ---")
    try:
        processor = TextProcessor(config)

        test_text_1 = "Hello world, this is number 123."
        test_text_2 = "Dr. Smith lives at 221B Baker St."
        test_text_3 = "This costs $5.99 or £10."

        print(f"\nRaw: '{test_text_1}'")
        ids_1, mask_1 = processor.process(test_text_1, phonemes=True)
        print(f"Phoneme IDs: {ids_1}")
        if processor.tokenizer and ids_1:
            print(f"Decoded: {processor.tokenizer.decode(ids_1)}")
        ids_1_txt, mask_1_txt = processor.process(test_text_1, phonemes=False)
        print(f"Text IDs: {ids_1_txt}")
        if processor.tokenizer and ids_1_txt:
            print(f"Decoded: {processor.tokenizer.decode(ids_1_txt)}")

        print(f"\nRaw: '{test_text_2}'")
        ids_2, mask_2 = processor.process(test_text_2, phonemes=True)
        print(f"Phoneme IDs: {ids_2}")
        if processor.tokenizer and ids_2:
            print(f"Decoded: {processor.tokenizer.decode(ids_2)}")

        print(f"\nRaw: '{test_text_3}'")
        ids_3, mask_3 = processor.process(test_text_3, phonemes=True)
        print(f"Phoneme IDs: {ids_3}")
        if processor.tokenizer and ids_3:
            print(f"Decoded: {processor.tokenizer.decode(ids_3)}")

    except Exception as e:
        print(f"ERROR during TextProcessor test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Text Processing Test Finished ---") 