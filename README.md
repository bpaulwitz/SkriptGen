# ScriptGen

## Roadmap
- [x] Data preprocessing
    - [x] Script to Tokens _(`word_tokens.py`, `sentence_tokens.py`)_
    - [x] Encoding words _(`word_tokens.py`)_
        - [x] Create Encoding
        - [x] Store Encoding
        - [x] Load Encoding
        - [x] Tokens to Encoding
    - [x] Encoding Sentences _(`sentence_tokens.py`)_
        - [x] Create Encoding
        - [x] Store Encoding
        - [x] Load Encoding
        - [x] Tokens to Encoding
    - [x] Dataset Class _(`dataset.py`)_
        - [x] Read sample
        - [x] Transform sample
            - [x] Rescale
            - [x] Numpy array to torch tensor
- [x] ScriptGen
    - [x] Image Encoder -> changed to ResNet as in PolyGen
    - [x] Transformer Decoder for script corpus
    - [x] Transformer Decoder for floating point values
- [x] Dataset
- [x] Create `setup.py` to install necessary libraries
