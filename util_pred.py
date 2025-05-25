"""
Utility
"""
import itertools
from tensorflow.keras.models import load_model
from datetime import datetime


def print_data(title, data_set, add_val=0, need_sort=False):
    """ print_data """
    print(title)
    cnt = 1
    for pre in data_set:
        p_temp = [ int(i) + add_val for i in pre ]
        if need_sort:
            p_temp.sort()
        print(f'{cnt} : {p_temp}')
        cnt += 1


def flat_data_with_sum(datas):
    """ flat_data_with_sum """
    return sum(datas, [])


def flat_data(datas):
    """ flat_data """
    return list(itertools.chain(*datas))


def get_frequency(datas: list) -> dict:
    """ get_frequency """
    frequency = {}
    for d in datas:
        if d in frequency:
            frequency[d] += 1
        else:
            frequency[d] = 1
    return frequency


def save_model(model):
    """ save_model """
    file_path = './' + f'{datetime.now()}' + '.h5'
    model.save(file_path)


def import_mode(file_path):
    """ import_model """
    return load_model(file_path)


def get_random_in_list(datas, num, use_shuffle=True) -> list:
    """ get_random_in_list """
    import random
    # Select a single random element
    # Shuffle the list in place
    random_element = []
    for n in range(num):
        if use_shuffle:
            random.shuffle(datas)
        choiced = random.choice(datas)
        choiced.sort()
        random_element.append(choiced)
    return random_element


def get_sorted_n_values(my_dict, n=5, is_reverse=True, with_key=False):
    """
    Returns the top N values from a dictionary.

    Args:
        my_dict (dict): The dictionary to extract values from.
        n (int): The number of top values to retrieve (default is 5).

    Returns:
        list: A list containing the top N values from the dictionary, 
              sorted in descending order.
              Returns an empty list if the dictionary is empty.
    """
    import itertools
    new_dict = {}
    if not my_dict:
        return {}, new_dict
    if with_key:
        sorted_values = sorted(my_dict.items(),
                               reverse=is_reverse)
    else:
        sorted_values = sorted(my_dict.items(),
                               key=lambda kv: kv[1],
                               reverse=is_reverse)
    for s in sorted_values:
        new_dict[s[0]] = s[1]
    return dict(itertools.islice(new_dict.items(), n)), new_dict


def list_item_count_to_dict(lists):
    """ list_item_count_to_dict """
    new_dict = {}
    for n in lists:
        if n in new_dict:
            new_dict[n] += 1
        else:
            new_dict[n] = 1
    return new_dict


def print_data_with_sort(title, data_set, add_val=0, need_sort=False):
    """ print_data_with_sort """
    print(title)
    cnt = 1
    for pre in data_set:
        pre.sort()
        p_temp = [ int(i) + add_val for i in pre ]
        if need_sort:
            p_temp.sort()
        print(f'{cnt} : {p_temp}')
        cnt += 1


def dict_key_count(target, datas) -> ():
    """ check_count """
    result = []
    cnt_dict = {}
    
    for data in datas:
        new1 = sorted(set(data).intersection(target))
        len_new1 = len(new1)
        try:
            cnt_dict[len_new1] += 1
        except:
            cnt_dict[len_new1] = 1
        result.append((data, new1))
    return cnt_dict, result


def print_list(title: str, datas: list):
    """ print_list """
    print(f'{title}')
    for data in datas:
        print(f'\t{data}')


def print_dict_list(title: str, datas: [], is_sort=False, is_reverse=False):
    """ print_dict_list """
    print(f'{title}')
    cnt = 1
    for data in datas:
        new_data = get_sorted_n_values(my_dict=data, n=len(data), with_key=True, is_reverse=is_reverse)
        print(f'\t{cnt}\'s : {new_data[1]}')
        cnt+=1


def print_title(title=""):
    """ print_title """
    print(f'{"*"*30}\n\t{title}\n{"*"*30}')
