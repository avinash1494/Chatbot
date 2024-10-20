import string
def process_string(input_str):
    # Remove newlines and specified punctuation except space and period
    input_str = input_str.replace('\n', ' ')  # Replace newlines with space
    input_str = ''.join([char for char in input_str if char not in string.punctuation or char in ['.', ' ']])

    # If the string exceeds 1500 characters, truncate it
    if len(input_str) > 1500:
        input_str = input_str[:1500]
    
    return input_str