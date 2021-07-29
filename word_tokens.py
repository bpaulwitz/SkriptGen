char_token_dictionary = {
    '_' : "_UNDERSCORE_", # always has to be in the first place!!
    '.' : "_DOT_",
    '/' : "_SLASH_",
    ',' : "_COMMA_",
    '*' : "_STAR_",
    '+' : "_PLUS_",
    '-' : "_MINUS_",
    '{' : "_CURLBRACKOPEN_",
    '}' : "_CURLBRACKCLOSE_",
    '[' : "_SQUAREBRACKOPEN_",
    ']' : "_SQUAREBRACKCLOSE_",
    '(' : "_BRACKOPEN_",
    ')' : "_BRACKCLOSE_",
    ':' : "_COLUMN_",
    ';' : "_SEMICOLON_",
    '?' : "_QUESTMARK_",
    '!' : "_EXCLMARK_",
    '<' : "_LESS_",
    '>' : "_GREATER_",
    '"' : "_QUOTMARK_",
    '\'' : "_BACKSLASH_",
    '%' : "_PERCENT_",
    '=' : "_EQUALS_",
    '\n' : "_NEWLINE_",
    '\t' : "_TABULATOR_",
    ' ' : "_SPACE_"
}

int_token_dictionary = {
    '1' : "__ONE__",
    '2' : "__TWO__",
    '3' : "__THREE__",
    '4' : "__FOUR__",
    '5' : "__FIVE__",
    '6' : "__SIX__",
    '7' : "__SEVEN__",
    '8' : "__EIGHT__",
    '9' : "__NINE__",    
}
token_SOS = "_STARTOFSENTENCE_"
token_EOS = "_ENDOFSENTENCE_"
token_PAD = "_PADDING_"
token_FLOAT = "_FLOAT_"

def is_float(word: str) -> bool:
    """
    check if a string is a floating point value
    @param word: string to check
    @return: True if string is float, False otherwise
    """
    try:
        float(word)
        return True
    except ValueError:
        return False

# see https://stackoverflow.com/questions/1265665/how-can-i-check-if-a-string-represents-an-int-without-using-try-except
def is_int(word: str) -> int:
    """
    check if a string is representing a int
    @param word: string to check
    @return: True if string is int, False otherwise
    """
    # unnecessary, becouse there is no '+' or '-' left in the script
    #if word[0] in ('-', '+'):
    #    return word[1:].isdigit()
    return word.isdigit()

def is_number(word: str) -> bool:
    """
    check if a string is a number
    @param word: string to check
    @return: True if string is float or int, False otherwise
    """

    return is_int(word) or is_float(word)

def read_script_no_comments(script_path: str) -> str:
    """
    reads python script content from path and removes lines with preceding '#'
    @param script_path: path of the script
    @return: script content without comments
    """
    script_content = ""
    with open(script_path, 'r') as script:
        lines = script.readlines()
        # read lines that do not start with an '#'
        for l in lines:
            if not l.lstrip().startswith('#'):
                script_content += l
    return script_content

def script_to_tokens(script_path: str, blank_floats: bool = True, batch_length: int = None) -> list:
    """
    convert script to a list of tokens
    @param script_path: path to the python script
    @param blank_floats: True: blank occuring floats in the script by the float token and False: write floats instead
    @param batch_length: if set, add padding tokens until the batch_length is reached
    @return: list of tokens
    """
    tokenized = []
    script = read_script_no_comments(script_path)
    script_char_tokenized = script
    
    # replace special characters with their respective tokens
    for char in char_token_dictionary:
        # '.' is in floats -> has to be done seperately
        if char != '.':
            script_char_tokenized = script_char_tokenized.replace(char, char_token_dictionary[char])

    # do the '.'
    index = 0
    while(script_char_tokenized.find('.', index) != -1):
        found = script_char_tokenized.index('.', index)
        # if the char before or after the dot is a int -> it is a dot in a floating point value
        if is_int(script_char_tokenized[found-1]) or is_int(script_char_tokenized[found+1]):
            index = found + 1
        # otherwise: replace this occurence of '.' by its token value
        else:
            script_char_tokenized = script_char_tokenized[:found] + char_token_dictionary['.'] + script_char_tokenized[found + 1:]
            # don't need to take the new length into account since there is no dot in the token value of dot
            index += 1
    
    # function to break up a string of an integer number to its tokens in int_token_dictionary
    def int_to_token(word: str) -> list:
        assert(is_int(word))
        tokens = []
        if word[0] in ('+', '-'):
            word = word[1:]
        for char in word:
            tokens.append(int_token_dictionary[char])
        return tokens
    
    # function to add a word to the tokenized list
    def add_word(word: str, tokenized_list: list):
        # if word is empty -> return
        if word == "":
            return

        # if the word is an int -> split it to its floats and add their tokens
        if is_int(word):
            tokenized_list += int_to_token(word)
            return

        # if we want to blank the floats by their tokens (which makes the whole tokenized script a list of discrete values)
        if blank_floats:
            # if it is a float
            if is_float(word):
                tokenized_list.append(token_FLOAT)
            # not a float
            else:
                tokenized_list.append(word)
        # add word without checking if it is a float
        else:
            tokenized_list.append(word)

    # tokenize script
    index = 0
    current_word = ""
    script_len = len(script_char_tokenized)
    while(index < script_len):
        current_char = script_char_tokenized[index]

        # if we detect the start of a token
        if current_char == '_':
            # save current word to list
            add_word(current_word, tokenized)
            current_word = ""

            # save token to list
            token_end = script_char_tokenized.find('_', index + 1, script_len)
            if token_end == -1:
                raise ValueError("ERROR: could not tokenize script " + script_path)
            else:
                tokenized.append(script_char_tokenized[index:token_end + 1])
                index = token_end + 1
        # if we 'stay' in a word which is not a character token
        else:
            current_word += current_char
            index += 1
    
    # if the last word is not empty, add it to list
    if current_word != "":
        add_word(current_word)

    # add start and end token and filter empty words -> https://stackoverflow.com/questions/3845423/remove-empty-strings-from-a-list-of-strings
    tokenized = [token_SOS] + list(filter(None, tokenized)) + [token_EOS]

    # if we have to fill it until a batch length
    if batch_length is not None:
        # difference of batch length and token length (minus one for the end token)
        diff = batch_length - len(tokenized)
        tokenized += [token_PAD] * diff

    return tokenized

def encode_script(file_path: str, encoding: dict, blank_floats: bool = True, batch_length: int = None) -> list:
    """
    encode a script with the given encoding set
    @param file_path: path to script (including file name)
    @param encoding: encoding dictionary
    @param blank_floats: True: replace occuring floats in the script by the encoded float-token and False: write floats instead
    @param batch_length: if set, add encoded padding tokens until the batch_length is reached
    @return: the list of encoded tokens (and floats if blank_floats == False)
    """
    encoded = []
    tokens = script_to_tokens(file_path, blank_floats=blank_floats, batch_length=batch_length)
    # if floats are blanked (tokens are only words and should all be in the encoding keys)
    if blank_floats:
        for t in tokens:
            encoded.append(encoding[t])
    # if floats are not blanked (tokens are mixed with actual floats)
    else:
        for t in tokens:
            # if token is key in encoding (a word)
            if t in encoding:
                encoded.append(encoding[t])
            # else if token is not in encoding (becouse it is a float)
            else:
                encoded.append(t)
    return encoded

def get_floats_encoded_script(encoded_blanked: list, encoded_with_floats: list, encoding: dict) -> list:
    """
    compare the blanked encoded version of a file with it's unblanked variant and retrieve the floating point values.
    @param encoded_blanked: encoded version of the script with blanked floats
    @param encoded_with_floats: encoded version of the same script with original floats
    @param encoding: encoding dictionary
    @return: the list of floats
    """
    floats = []
    index = 0
    # value of encoded float-token (initialize with -1 since all values of the dictionary are >= 0)
    encoded_float = -1
    if token_FLOAT in encoding:
        encoded_float = encoding[token_FLOAT]

    while index < len(encoded_blanked):
        encoded_value = encoded_blanked[index]
        # if the encoded token at the index is a float
        if  encoded_value == encoded_float:
            # add to list
            floats.append(encoded_with_floats[index])
        # else: make sure that the scripts are equal, just to minimize the chance of accidently using wrong parameters
        else:
            assert(encoded_blanked[index] == encoded_with_floats[index])
        index += 1
    return floats

def create_encoding(script_paths: list) -> dict:
    """
    create an encoding based on a list of scripts
    @param script_paths: list of path to scripts where to create the encoding from
    @return: encoding dictionary
    """
    encoding_tokens = []
    encoding = {}

    for path in script_paths:
        tokens = script_to_tokens(path)
        
        # add all tokens which are not yet in the list
        for t in tokens:
            if t not in encoding_tokens:
                encoding_tokens.append(t)
        
    # fill dictionary with (value : index) from list, so every token is represented by a word
    for index, token in enumerate(encoding_tokens):
        encoding[token] = index
    return encoding

def save_encoding(encoding: dict, file_path: str):
    """
    save the encoding dict to file
    @param encoding: encoding dictionary
    @param file_path: path (including file name) where the encoding should be saved
    """
    file_content = ""
    for token in dict(sorted(encoding.items(), key=lambda item: item[1])):
        file_content += token + " : " + str(encoding[token]) + '\n'

    with open(file_path, 'w+') as encoding_file:
        # write everything to file (except the last new line)
        encoding_file.write(file_content[:-1])

def load_encoding(file_path: str) -> dict:
    """
    load an encoding from a file path
    @param file_path: path to the encoding file
    @return: the encoding dictionary
    """
    encoding = {}
    with open(file_path, 'r') as encoding_file:
        for line in encoding_file.readlines():
            token_value = line.split(' : ')
            encoding[token_value[0]] = int(token_value[1])

    return encoding

def decoding_from_encoding(encoding: dict) -> dict:
    """
    converts an encoding dictionary to a decoding dictionary
    @return: decoding dictionary
    """
    decoding = {}
    for token in encoding:
        decoding[encoding[token]] = token
    return decoding

