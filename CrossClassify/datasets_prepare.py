# -*- coding: UTF-8 -*-
"""
Filename: img_pre_process.py
Function:
    1.crop picture into square
    2.resize all the picture to 320*320 pixels in this dic
    3.get 4 times number pictures by rotating
Author: To_Fourier
Created Time: 2020.07.11.10.26
Last Modified: 2020.07.21.07.21
"""
from PIL import Image
import os


def img_resize(img, size):
    goal_size = [size, size]
    img = img_crop(img)
    width = int(goal_size[0])
    height = int(goal_size[1])
    img_new = img.resize((width, height), Image.ANTIALIAS)
    return img_new


def img_crop(img):
    img_size = img.size
    if img_size[0] <= img_size[1]:
        new_size = img_size[0]
        left = 0
        up = (img_size[1] - new_size) / 2
    else:
        new_size = img_size[1]
        left = (img_size[0] - new_size) / 2
        up = 0
    width = new_size
    depth = new_size

    img_new = img.crop((left, up, left + width, up + depth))
    return img_new


def img_rotate(img, angle):
    img_new = img.rotate(angle)
    return img_new


def img_transpose(img, horizontal):
    if horizontal:
        img_new = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        img_new = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img_new


if __name__ == '__main__':
    goal_size_set = 256
    child_paths = ['A1', 'A2', 'A3',
                   'B1', 'B2', 'B3',
                   'C1', 'C2', 'C3', 'C4',
                   'D1', 'D2', 'D3', 'D4',
                   'E1', 'E2', 'E3',
                   'F1', 'F2', 'F3',
                   'G1', 'G2', 'G3',
                   'H1', 'H2', 'H3',
                   'I1', 'I2', 'I3', 'I4']
    for child_path in child_paths:
        count = 0
        input_path = 'C:\\Users\\44375\\Desktop\\datasets\\' + child_path
        output_path = 'C:\\Users\\44375\\Desktop\\CrossClassifyBIG\\Datasets_TRAIN' + \
                      str(goal_size_set) + '\\' + child_path
        files = os.listdir(input_path)
        os.chdir(input_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for file in files:
            if os.path.isfile(file):
                img = Image.open(file)

                img_resized = img_resize(img, goal_size_set)
                img_resized.save(os.path.join(output_path, '01_' + str(count) + '.jpg'))

                img_rotate_90 = img_rotate(img_resized, 90)
                img_rotate_90.save(os.path.join(output_path, '02_' + str(count) + '.jpg'))

                img_rotate_180 = img_rotate(img_resized, 180)
                img_rotate_180.save(os.path.join(output_path, '03_' + str(count) + '.jpg'))

                img_rotate_270 = img_rotate(img_resized, 270)
                img_rotate_270.save(os.path.join(output_path, '04_' + str(count) + '.jpg'))

                img_resized_trans_horizontal = img_transpose(img_resized, True)
                img_resized_trans_horizontal.save(os.path.join(output_path, '05_' + str(count) + '.jpg'))

                img_rotate_90_trans_horizontal = img_transpose(img_rotate_90, True)
                img_rotate_90_trans_horizontal.save(os.path.join(output_path, '06_' + str(count) + '.jpg'))

                img_rotate_180_trans_horizontal = img_transpose(img_rotate_180, True)
                img_rotate_180_trans_horizontal.save(os.path.join(output_path, '07_' + str(count) + '.jpg'))

                img_rotate_270_trans_horizontal = img_transpose(img_rotate_270, True)
                img_rotate_270_trans_horizontal.save(os.path.join(output_path, '08_' + str(count) + '.jpg'))

                img_resized_trans_vertical = img_transpose(img_resized, False)
                img_resized_trans_vertical.save(os.path.join(output_path, '09_' + str(count) + '.jpg'))

                img_rotate_90_trans_vertical = img_transpose(img_rotate_90, False)
                img_rotate_90_trans_vertical.save(os.path.join(output_path, '10_' + str(count) + '.jpg'))

                img_rotate_180_trans_vertical = img_transpose(img_rotate_180, False)
                img_rotate_180_trans_vertical.save(os.path.join(output_path, '11_' + str(count) + '.jpg'))

                img_rotate_270_trans_vertical = img_transpose(img_rotate_270, False)
                img_rotate_270_trans_vertical.save(os.path.join(output_path, '12_' + str(count) + '.jpg'))

                count += 1

        print('Process %s Finished' % child_path)
