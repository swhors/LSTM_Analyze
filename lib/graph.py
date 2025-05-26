import itertools
import matplotlib.pyplot as plt


def draw_data_graph(data_number_set, verb=True):
    """ draw_data_graph """
    # flated_data = flat_data(data_number_set)
    flated_data = itertools.chain.from_iterable(data_number_set)
    if verb:
        print(f'{flated_data}')
    frequency = get_frequency(flated_data)
    frequency_sorted = dict(sorted(frequency.items()))
    if verb:
        print(frequency_sorted)
    x_axis = list(frequency_sorted.keys())
    y_axis = list(frequency_sorted.values())
    plt.bar(x_axis, y_axis)
    plt.xlabel("nums")
    plt.ylabel("frequency")
    plt.title("frequency")
    plt.show()
    if verb:
        print_data(title='wanted_data',
                   data_set=[[5, 12, 24, 26, 39, 42, 20]])
