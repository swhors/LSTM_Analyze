import os


def get_sorted_n_values(my_dict, n=5, is_reverse=True, with_key=False):
    """ get_sorted_n_values """
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


fd = open('sel.txt', 'r')
line = fd.readline()
nums = []
while line:
    num_strs = line.split(",")
    line = fd.readline()
    nums += [int(i) for i in num_strs]
fd.close()

nums_dict = {}
for n in nums:
    if n in nums_dict:
        nums_dict[n] += 1
    else:
        nums_dict[n] = 1

sorted_dict1, _ = get_sorted_n_values(nums_dict, len(nums_dict), with_key=False)
sorted_dict2, _ = get_sorted_n_values(nums_dict, len(nums_dict), with_key=True)

print(sorted_dict1)
print(sorted_dict2)
