# ScriptGen

## Roadmap
- [ ] Data preprocessing
    - [x] Script to Tokens _(`word_tokens.py`, `sentence_tokens.py`)_
    - [x] Encoding words _(`word_tokens.py`)_
        - [x] Create Encoding
        - [x] Store Encoding
        - [x] Load Encoding
        - [x] Tokens to Encoding
    - [ ] Encoding Sentences _(`sentence_tokens.py`)_
        - [ ] Create Encoding
        - [ ] Store Encoding
        - [ ] Load Encoding
        - [ ] Tokens to Encoding
    - [x] Dataset Class _(`dataset.py`)_
        - [x] Read sample
        - [x] Transform sample
            - [x] Rescale
            - [x] Numpy array to torch tensor
- [ ] ScriptGen
    - [ ] Image Encoder (based on ViT)
    - [ ] Transformer Decoder for script corpus
    - [ ] Transformer Decoder for floating point values
- [ ] Dataset
- [ ] Create `setup.py` to install necessary libraries
