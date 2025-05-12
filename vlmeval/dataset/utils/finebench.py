from ...smp import *
from .multiple_choice import extract_answer_from_item
import numpy as np
import re

FAIL_MSG = 'Failed to obtain answer via API.'


def get_dimension_rating(data_path):
    data = load(data_path)
    result_board = {}
    for idx, item in data.iterrows():
        if item['action_type'] not in result_board:
            result_board[item['action_type']] = [0, 0]
        result_board[item['action_type']][1] += 1
        if item['score']:
            result_board[item['action_type']][0] += 1

    correct = 0
    total = 0
    for key, value in result_board.items():
        correct += value[0]
        total += value[1]
        result_board[key].append(f'{value[0] / value[1] * 100 :.2f}%')

    result_board['overall'] = [correct, total, f'{correct / total * 100 :.2f}%']

    return result_board


def extract_option(model, input_item, dataset_name):
    options = input_item['question'].split('\n')[1:]
    for id, option in enumerate(options):
        option_id = chr(ord('A') + id) + '.'
        if option.find(option_id) >= 0:
            input_item[chr(ord('A') + id)] = option[option.find(option_id) + len(option_id):].strip('. \n')
    return extract_answer_from_item(model, input_item, dataset_name)['opt']


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is'
        'The correct option is',
        'Best answer:'
        'Best option:',
        'Answer:',
        'Option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCD]', s):
        return ''
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ''
    return matches[0]
