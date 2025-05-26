"""
Json_util.py
"""
import json
import os
from datetime import datetime


def write_json(trial, matched_count, matched_list, append=False):
    """ write_json """
    suffix = datetime.now().strftime('%y%m%d_%H')
    file_name = f"matched_{suffix}.json"
    matched_dict = {}
    if append:
        if os.path.isfile(file_name):
            with open(file_name, "r") as file:
                matched_dict = json.load(file)
                file.close()
    counts, datas = change_matched_info(matched_count, matched_list)
    if len(counts) > 0 and len(datas) > 0:
        matched_dict[trial] = {}
        matched_dict[trial]["counts"] =  counts
        matched_dict[trial]["datas"] = datas
        print(f'matched_dict = {matched_dict}')
        with open(file_name, "w") as file:
            json.dump(matched_dict, file, indent=4)
