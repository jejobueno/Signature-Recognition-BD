from lxml import etree
from lxml.cssselect import CSSSelector
import os
import os.path as op
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from utils.YOLOv5_format import transform_to_YOLOv5_form
from utils.rescale_data import scale_image

# This variable is going to count the documents with signatures
from utils.split_train_val_test import split_data

count_sig = 0
# CSSSelector to look for the 'DLSignature' object and the 'DL_PAGE'
sel = CSSSelector('[gedi_type="DLSignature"]')
selPage = CSSSelector('[gedi_type="DL_PAGE"]')

# Initializing lists to save relevant info from each xml
cols = []
rows = []
heights = []
file_names = []
widths = []
target = []
page_widths = []
page_heights = []
author_ids = []
overlapped = []

# Loop  to walk around all xml files from the 'train_xml' folder
for root, folders, files in os.walk("data/train_xml"):
    for file in files:
        tree = etree.parse(op.join(root, file))
        xml_root = tree.getroot()
        if sel(xml_root):
            count_sig += 1
            target.append(1)
            for elem in selPage(xml_root):
                page = dict(elem.items())
            for elem in sel(xml_root):
                signature = dict(elem.items())
                file_names.append(os.path.basename(file).removesuffix('.xml') + '.tif')
                page_heights.append(page['height'])
                page_widths.append(page['width'])
                cols.append(signature['col'])
                rows.append(signature['row'])
                heights.append(signature['height'])
                widths.append(signature['width'])
                author_ids.append(signature['AuthorID'])
                overlapped.append(signature['Overlapped'])
        else:
            target.append(0)

print(f"Found {count_sig} images with a signature")

# Create dataframe with the extracted data of the signatures from xml
df_bbox = pd.DataFrame(zip(file_names, cols, rows, heights, widths, page_heights, page_widths, author_ids, overlapped),
                       columns=['file', 'x', 'y', 'height', 'width', 'page_heights', 'page_widths', 'author_id',
                                'overlapped'])
# Second data frame to verify training model afterwards
df_signatures = pd.DataFrame(zip(files, target), columns=['file', 'hasSignature'])

# Rescaling the data to faster processing
df_bbox_scaled_data = scale_image(df_bbox)

scaled_data = list(zip(*df_bbox_scaled_data))

df_bbox['jpg_filename'] = scaled_data[0]
df_bbox['x_scaled'] = scaled_data[1]
df_bbox['y_scaled'] = scaled_data[2]
df_bbox['w_scaled'] = scaled_data[3]
df_bbox['h_scaled'] = scaled_data[4]
df_bbox['page_height_scaled'] = scaled_data[5]
df_bbox['page_width_scaled'] = scaled_data[6]

# Reformatting the coordinates to YOLOv5 format
df_bbox_YOLOv5_data = transform_to_YOLOv5_form(df_bbox)

print(f'###### NUMBER OF EXAMPLES {df_bbox.shape[0]}')

samples = df_bbox.jpg_filename.unique()

df_train, df_val = train_test_split(samples, test_size=0.1, random_state=42, shuffle=True)

print(df_train.shape[0], df_val.shape[0])
split_data(df_bbox, df_val, 'val')
split_data(df_bbox, df_train, 'train')

# Saving the dataframes into csv files
df_bbox.to_csv('data/signatures_bbox_df.csv', index=False)
df_signatures.to_csv('data/signatures_df.csv', index=False)

print("No. of TOTAL training images", len(os.listdir('data/train')))
print("No. of TOTAL signed training images", len(os.listdir('data/YOLOv5_formatted_data/images/jpg_scaled_images')))
#
print("No. of Training images", len(os.listdir('data/YOLOv5_formatted_data/images/train')))
print("No. of Training labels", len(os.listdir('data/YOLOv5_formatted_data/labels/train')))

print("No. of val images", len(os.listdir('data/YOLOv5_formatted_data/images/val')))
print("No. of val labels", len(os.listdir('data/YOLOv5_formatted_data/labels/val')))

