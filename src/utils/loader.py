import os
import io
import glob
import torch
import random
import string
import unicodedata

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)


def unicode_to_ascii(s):
    """
    takes a unicode string and returns an ascii string
    """
    return ''.join(
        ch for ch in unicodedata.normalize('NFD', s)
        if unicodedata.category(ch) != 'Mn'
        and ch in ALL_LETTERS
    )
    

def load_data():
    """
    Load the data from the data folder
    """
    names_dict = {}
    catagoires = []
    
    
    def find_files(path):
        """"
        Find all the files in the path
        """
        return glob.glob(path)
    
    
    def read_lines(filename):
        """
        Read the lines from the file
        """
        with io.open(filename, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]
    
    
    for filename in find_files('../../data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        catagoires.append(category)
        
        lines = read_lines(filename)
        names_dict[category] = lines
        
    return names_dict, catagoires


def letter_to_index(letter):
    """
    Convert the letter to the index
    """
    return ALL_LETTERS.find(letter)


def letter_to_tensor(letter):
    """
    Convert the letter to the tensor
    """
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line):
    """
    Convert the line to the one hot tensor
    """
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


def random_training_example(names_dict, categories):
    """
    Get a random training example
    """
    category = random.choice(categories)
    line = random.choice(names_dict[category])
    category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor
    