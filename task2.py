"""
Character Detection
(Due date: March 8th, 11: 59 P.M.)

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
and the functions you implement in task1.py are of great help.

Hints: You might want to try using the edge detectors to detect edges in both the image and the template image,
and perform template matching using the outputs of edge detectors. Edges preserve shapes and sizes of characters,
which are important for template matching. Edges also eliminate the influence of colors and noises.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os

import utils
from task1 import *   # you could modify this line


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="./data/characters.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def detect(img, template, threshold=0.7):
    """Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
    # TODO: implement this function.
    ccoeff_normed_mat = get_ccoeff_normed(img, template)
    #Get the coordinates
    #Inverted as the x maps to columns and y maps to rows from pixel coords to matrix
    coordinates = [(y,x) for x in range(len(ccoeff_normed_mat)) \
        for y in range(len(ccoeff_normed_mat[0])) if ccoeff_normed_mat[x][y] >= threshold]    
    return coordinates


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)

#Reference - https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html
def get_ccoeff_normed(img, template):
    
    arr_element_sub = lambda arr, val: [arr_val - val for arr_val in arr]
    element_squared_sum = lambda arr: sum(z**2 for z in arr)
    sum_normed = lambda arr, const: sum(arr) / const
    sum_of_prods = lambda arr1, arr2: sum([a1_val * a2_val for a1_val, a2_val in zip(arr1, arr2)])
    flatten = lambda arr : [arr_val for arr_row in arr for arr_val in arr_row]

    w ,h =  len(template), len(template[0]),
    wh_prod = w*h*1.0
    
    ccoeff_normed = [[0.0 for _ in range(len(img[0]) - h + 1)] for _ in range(len(img) - w + 1)]

    #Is same for every sliding window as template doesn't change
    template = flatten(template)
    t_sum_normed = sum_normed(template, wh_prod)
    t_dash = arr_element_sub(template, t_sum_normed)
    t_dash_sum_squared = element_squared_sum(t_dash)

    #Hack for zeros (make sure the denomintor doesn't become zero)
    if(t_dash_sum_squared == 0):
        t_dash_sum_squared = 1e-6

    def get_ccoeff_normed_val(i, j,):

        #Flatten the image
        im = [img[x][y] for x in range(i, i + w) for y in range(j, j + h)]

        i_sum_normed = sum_normed(im, wh_prod)
        i_dash = arr_element_sub(im, i_sum_normed)
        i_dash_sum_squared = element_squared_sum(i_dash)
        #Hack for zeros (make sure the denomintor doesn't become zero)
        if i_dash_sum_squared == 0:
            i_dash_sum_squared = 1e-6
                    
        numerator = sum_of_prods(t_dash, i_dash)
        denominator = np.sqrt(t_dash_sum_squared * i_dash_sum_squared)
        
        return numerator/denominator
    
    for x in range(len(ccoeff_normed) - w):
        for y in range(len(ccoeff_normed[0]) - h):
            ccoeff_normed[x][y] = get_ccoeff_normed_val(x, y)
    return ccoeff_normed


def main():
    args = parse_args()

    img = read_image(args.img_path)
    template = read_image(args.template_path)
    #Added begins
    show_image(np.array(img), 100000)
    #Added ends

    coordinates = detect(img, template)

    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
    main()
