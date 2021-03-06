# import tokens
from word_tokens import char_token_dictionary, token_FLOAT, token_EOS, token_PAD, token_SOS
# import important functions
from word_tokens import is_int, decoding_from_encoding
# regular expressions
import re

# precompile regex to find numbers
# taken from https://stackoverflow.com/questions/45001775/find-all-floats-or-ints-in-a-given-string/45001796
# [sign] (digit+ [dot] digit+ | dot digit+) [exp] [sign] digit+
re_find_numbers = re.compile("[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?")

def read_script_lines_no_comments(script_path: str) -> str:
    """
    reads python script lines from path and removes lines with preceding '#'
    @param script_path: path of the script
    @return: script lines without comments
    """
    uncommented_lines = []
    with open(script_path, 'r') as script:
        lines = script.readlines()
        # read lines that do not start with an '#'
        for l in lines:
            if not l.lstrip().startswith('#'):
                uncommented_lines.append(l)
    return uncommented_lines

def script_to_sentence_tokens(script_path: str):
    """
    convert a script to its sentence tokens. That means that each line gets tokenized and that the floating point numbers are collected together with the 
    tokenized lines.
    @param script_path: path to the script
    @return tokenized lines and list of floating point numbers
    """
    def strip_line_from_numbers(line: str):
        numbers = re_find_numbers.findall(line)
        floating_numbers = []
        for n in numbers:
            # skip the integer numbers
            if not '.' in n:
                continue
            else:
                floating_numbers.append(n)
            # check that number is not included in any other numbers and then replace it
            index = 0
            # check if char before float in line is not a number
            start_at = 0
            while (True):
                index = line.find(n, start_at)
                if (index > -1):
                    # if this number is in a bigger number string (preceeding or succeeding digits)
                    pre_digits = True
                    suf_digits = True

                    # there are characters following the number
                    if len(line) > index + len(n):
                        # following character is not a digit
                        if not line[index + len(n)].isdigit():
                            suf_digits = False
                    # there are preceeding characters
                    if index > 0:
                        # preceeding character is not a digit
                        if not line[index - 1].isdigit():
                            pre_digits = False

                    # number not in bigger number -> can replace
                    if not (pre_digits or suf_digits):
                        # replace floats by Float-Token
                        line = line[:index] + token_FLOAT + line[index + len(n):]

                    start_at = index + len(n)
                else:
                    break

        # replace the '\n' character by the newline-token
        line = line.replace('\n', char_token_dictionary['\n'])

        return (line, floating_numbers)

    uncommented_lines = read_script_lines_no_comments(script_path)

    # make sure that the last line also ends with an \n
    if not uncommented_lines[-1].endswith('\n'):
        uncommented_lines[-1] = uncommented_lines[-1] + '\n'

    all_numbers = []
    tokenized_script_lines = []
    for ul in uncommented_lines:
        line, numbers = strip_line_from_numbers(ul)
        tokenized_script_lines.append(line)
        all_numbers += numbers

    return tokenized_script_lines, all_numbers

def create_sentence_encoding(script_paths: list) -> dict:
    """
    create an sentence encoding based on a list of scripts
    @param script_paths: list of path to scripts where to create the encoding from
    @return: encoding dictionary, maximum length of encoded scripts and maximum length of floating point numbers
    """
    encoding_tokens = [token_SOS, token_PAD, token_EOS]
    encoding = {}
    max_length_encoding = 0
    max_length_floats = 0

    for path in script_paths:
        lines, nbrs = script_to_sentence_tokens(path)
        
        # add all tokens which are not yet in the list
        for l in lines:
            if l not in encoding_tokens:
                encoding_tokens.append(l)

        # check if maximum length of encoding or the floating point numbers increased (additional space for start and end token)
        if len(lines) + 2 > max_length_encoding:
            max_length_encoding = len(lines) + 2
        if len(nbrs) + 2 > max_length_floats:
            max_length_floats = len(nbrs) + 2
        
    # fill dictionary with (value : index) from list, so every token is represented by an index
    for index, token in enumerate(encoding_tokens):
        encoding[token] = index

    return encoding, max_length_encoding, max_length_floats

def encode_sentence_script(file_path: str, encoding: dict, batch_length_skript: int = None, batch_length_numbers: int = None):
    """
    encode a script with the given sentence encoding set
    @param file_path: path to script (including file name)
    @param encoding: encoding dictionary
    @param batch_length_skript: if set, add encoded padding tokens to skript until the batch_length_skript is reached
    @param batch_length_numbers: if set, add encoded padding tokens to number list until the batch_length_numbers is reached
    @return: the list of encoded tokens and the list of the numbers
    """
    encoded = [encoding[token_SOS]]
    tokens, numbers = script_to_sentence_tokens(file_path)

    for t in tokens:
        encoded.append(encoding[t])

    # fill until batch_length if specified
    if batch_length_skript is not None and len(encoded) + 1 < batch_length_skript:
        additional_padding_len = batch_length_skript - len(encoded) - 1
        encoded += [encoding[token_PAD]] * additional_padding_len
    if batch_length_numbers is not None and len(numbers) + 1 < batch_length_numbers:
        additional_padding_len = batch_length_numbers - len(numbers) - 1
        numbers += [0] * additional_padding_len
    
    encoded.append(encoding[token_EOS])
    
    return encoded, numbers

def save_sentence_encoding(encoding: dict, max_len_encoding: int, max_len_floats: int, file_path: str):
    """
    save the encoding dict, the maximum length of an encoded script and the maximum length of floats for a script to file
    @param encoding: encoding dictionary
    @param max_len_encoding: maximum length of the encoded script
    @param max_len_floats: maximum length of the floating point list for the encoded script
    @param file_path: path (including file name) where the encoding should be saved
    """
    file_content = "max_len_encoding -- " + str(max_len_encoding) + "\nmax_len_floats -- " + str(max_len_floats) + "\n\n"
    for token in dict(sorted(encoding.items(), key=lambda item: item[1])):
        file_content += token + " : " + str(encoding[token]) + '\n'

    with open(file_path, 'w+') as encoding_file:
        # write everything to file (except the last new line)
        encoding_file.write(file_content[:-1])

def load_sentence_encoding(file_path: str):
    """
    load an encoding from a file path
    @param file_path: path to the encoding file
    @return: the encoding dictionary, maximum length of an encoded script and maximum length of a float list for an encoded script
    """
    encoding = {}
    max_len_encoding = None
    max_len_floats = None
    with open(file_path, 'r') as encoding_file:
        lines = encoding_file.readlines()
        max_len_encoding = int(lines[0].split(' -- ')[1])
        max_len_floats = int(lines[1].split(' -- ')[1])
        for line_number in range(3, len(lines)):
            line = lines[line_number]
            token_value = line.split(' : ')
            encoding[token_value[0]] = int(token_value[1])

    return encoding, max_len_encoding, max_len_floats

def decode_encoded_script(encoded_script: list, numbers: list, encoding: dict) -> str:
    """
    decode an encoded script
    @param encoded_script: list of encoded tokens (might be the output of SkriptGen)
    @param numbers: list of numbers that need to be embedded in the script at the correct positions
    @param encoding: the encoding dictionary that is used to decode the script
    @return: the decoded script
    """
    # confirm token_SOS at the beginning
    assert(encoded_script[0] == encoding[token_SOS])
    #assert(encoded_script[-1] == encoding[token_EOS])

    # create decoding
    decoding = decoding_from_encoding(encoding)

    # decode encoded_script
    decoded_lines = []
    for t in encoded_script:
        decoded_token = decoding[t]
        if decoded_token != token_SOS and decoded_token != token_EOS and decoded_token != token_PAD:
            decoded_lines.append(decoding[t])

    decoded_script = ''.join(decoded_lines)

    # replace newline-tokens by actual newlines
    decoded_script = decoded_script.replace(char_token_dictionary['\n'], '\n')

    # replace floats
    for n in numbers:
        decoded_script = decoded_script.replace(token_FLOAT, str(n), 1)

    return decoded_script