"""
graph_util.py

** PIL을 사용하기 위해서는 imagemagick이나 eog가 설치 되어 있어야 합니다.
    sudo apt-get install eog
        or
    sudo apt-get install imagemagick
"""
import itertools
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
from lib.metrics_loader import load_data_by_version


def draw_data_graph(title,
                    x_data,
                    y_data,
                    x_label,
                    y_label,
                    width=0,
                    height=0,
                    file_name="",
                    is_bar=True,
                    verb=False):
    """ draw_data_graph """
    if is_bar:
        plt.bar(x_data, y_data)
    else:
        plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if len(file_name) > 0:
        plt.legend(loc=0)
        plt.savefig(file_name)
    if width > 0 and height > 0:
        plt.figure(figsize=(width,height))
    plt.show()


def draw_datas_graph(title,
                     x_y_datas,
                     x_label,
                     y_label,
                     width=0,
                     height=0,
                     file_name="",
                     is_bar=True,
                     verb=False):
    """ draw_data_graph """
    for x_y_data in x_y_datas:
        if is_bar:
            plt.bar(x_y_data[0], x_y_data[1])
        else:
            plt.plot(x_y_data[0], x_y_data[1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if len(file_name) > 0:
        plt.legend(loc=0)
        plt.savefig(file_name)
    if width > 0 and height > 0:
        plt.figure(figsize=(width,height))
    plt.show()


def draw_data_graph_v2(title,
                       version,
                       x_data,
                       y_datas,
                       x_label,
                       y_label,
                       colors=[],
                       width=-1,
                       height=-1,
                       is_show=True,
                       is_save=False,
                       show_label=True,
                       title_as_filename=False,
                       verb=False):
    """ draw_data_graph_v2 """
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    for i in range(len(y_datas)):
        if i < len(colors):
            if show_label:
                plt.plot(x_data, y_datas[i], color=colors[i], label=f"model_{i+1}")
            else:
                plt.plot(x_data, y_datas[i], color=colors[i])
        else:
            if show_label:
                plt.plot(x_data, y_datas[i], label=f"model_{i+1}")
            else:
                plt.plot(x_data, y_datas[i])
    if title_as_filename:
        file_name = title + ".png"
    else:
        file_name=f"v_{version:04}.png"
    plt.legend(loc=0)
    if width > 0 and height > 0:
        plt.figure(figsize=(width,height))
    if is_save:
        plt.savefig(file_name)
    if is_show:
        plt.show()


# load data and draw graph
def draw_graph_with_metric(version=0, is_show=True, is_save=False, verbose=False):
    """ draw_graph_with_metric """
    model_metrics = load_data_by_version(version=version, verbose=False)
    colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple']
    draw_data_graph_v2(title=f"rounds and checked_len v_{version}",
                       version=version,
                       x_data=model_metrics[5]['rounds'],
                       y_datas=[model_metrics[1]['checked_size'],
                                model_metrics[2]['checked_size'],
                                model_metrics[3]['checked_size'],
                                model_metrics[4]['checked_size'],
                                model_metrics[5]['checked_size']],
                       x_label="round",
                       y_label="checked_size",
                       colors=colors,
                       is_show=is_show,
                       is_save=is_save
                       )


def load_image(file_name, base_height=500):
    """ load_image """
    try:
        img = Image.open(file_name)
        if base_height > 0:
            hpercent = (base_height/float(img.size[1]))
            wsize = int((float(img.size[0])*float(hpercent))) 
            new_img = img.resize((wsize, base_height))
            display(new_img)
        else:
            display(img)
    except Exception as e:
        print(e)


def load_all_image(base_path="./", verbose=True, base_height=500):
    """ load_all_image """
    import os
    file_list = os.listdir(base_path)
    file_list_png = [file for file in file_list if file.endswith(".png")]
    if verbose:
        print ("file_list: - before sort - {}".format(file_list_png))
    file_list_png.sort()
    if verbose:
        print ("file_list: - after soft - {}".format(file_list_png))
    for idx, name in enumerate(file_list_png):
        print(f'{idx} : {name}')        
        load_image(file_name=name, base_height=base_height)
