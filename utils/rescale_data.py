import os

import cv2

BASE_DIR = os.path.join('data', 'train')

NEW_DIR = os.path.join('data', 'images', 'YOLOv5_formatted_data', 'images', 'jpg_scaled_images')


def scale_image(df):
    df_new = []
    filename = df.file
    X, Y, W, H = map(int, df.x), map(int, df.y), map(int, df.width), map(int, df.height)
    for file, x, y, w, h in zip(filename, X, Y, W, H):
        image_path = os.path.join(BASE_DIR, file)
        img = cv2.imread(image_path, 1)
        page_height, page_width = img.shape[:2]
        max_height = 640
        max_width = 480

        # computes the scaling factor
        if max_height < page_height or max_width < page_width:
            scaling_factor = max_height / float(page_height)
            if max_width / float(page_width) < scaling_factor:
                scaling_factor = max_width / float(page_width)
            # scale the image with the scaling factor
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        jpg_filename = file[:-4] + '.jpg'
        new_file_path = os.path.join(NEW_DIR, jpg_filename)
        cv2.imwrite(new_file_path, img)  # write the scales image

        # save new page height and width
        page_height, page_width = page_height * scaling_factor, page_width * scaling_factor
        # compute new x, y, w, h coordinates after scaling
        x, y, w, h = int(x * scaling_factor), int(y * scaling_factor), int(w * scaling_factor), int(h * scaling_factor)
        row = [jpg_filename, x, y, w, h, page_height, page_width]
        df_new.append(row)
    return df_new
