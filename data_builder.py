import os
import numpy as np


import glob2

def txt_concatenator(DIR_PATH, OUTPUT_PATH):
    """
    Joins all text files in a directory into one
    :param DIR_PATH: Directory to text files
    :param OUTPUT_PATH: Path to output file
    :return:
    """

    filenames = glob2.glob(DIR_PATH + '/' +'*.txt')  # list of all .txt files in the directory
    filenames.sort()

    with open(OUTPUT_PATH, 'w') as f:
        for file in filenames:
            with open(file) as infile:
                f.write(infile.read()+'\n')

# print(filenames)

DIR_PATH_ENGLISH = 'data/questions/english'
OUTPUT_ENGLISH_PATH = 'data/questions/english.txt'

DIR_PATH_CYPHER = 'data/questions/cypher'
OUTPUT_CYPHER_PATH = 'data/questions/cypher.txt'


txt_concatenator(DIR_PATH_ENGLISH, OUTPUT_ENGLISH_PATH)
txt_concatenator(DIR_PATH_CYPHER, OUTPUT_CYPHER_PATH)

