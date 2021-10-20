import os

# Apply function
import shutil

import numpy as np
import pandas as pd

BASE_DIR = os.path.join('data', 'YOLOv5_formatted_data')

SCR_IMG_PATH = os.path.join(BASE_DIR, 'images')
SCR_LAB_PATH = os.path.join(BASE_DIR, 'labels')


def split_data(df: pd.DataFrame, dir: str):
    DIR_IMG_PATH = os.path.join(SCR_IMG_PATH, dir)
    DIR_LAB_PATH = os.path.join(SCR_LAB_PATH, dir)

    for filename in df.jpg_filename.to_list():
        yolo_list = []

        for _, row in df[df.jpg_filename == filename].iterrows():
            yolo_list.append([0, row.x_center_norm, row.y_center_norm, row.width_norm, row.height_norm])

        yolo_list = np.array(yolo_list)
        txt_filename = os.path.join(DIR_LAB_PATH, str(row.jpg_filename.split('.')[0]) + ".txt")
        # Save the .img & .txt files to the corresponding train and validation folders
        np.savetxt(txt_filename, yolo_list, fmt=["%d", "%f", "%f", "%f", "%f"])
        shutil.copyfile(os.path.join(SCR_IMG_PATH, 'jpg_scaled_images', row.jpg_filename), os.path.join(DIR_IMG_PATH, row.jpg_filename))
