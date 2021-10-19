from lxml import etree
from lxml.cssselect import CSSSelector
import os
import os.path as op
import pandas as pd
import joblib

count_sig = 0
sel = CSSSelector('[gedi_type="DLSignature"]')

cols = []
rows = []
heights = []
file_names = []
widths = []
target = []

for root, folders, files in os.walk("data/train_xml"):
    for file in files:
        tree = etree.parse(op.join(root, file))
        print(tree)
        xml_root = tree.getroot()
        if sel(xml_root):
            count_sig += 1
            target.append(1)
            for elem in sel(xml_root):
                signature = dict(elem.items())
                file_names.append(os.path.basename(file))
                cols.append(signature['col'])
                rows.append(signature['row'])
                heights.append(signature['height'])
                widths.append(signature['width'])
        else:
            target.append(0)

print(f"Found {count_sig} XML files with a signature")

df_bbox = pd.DataFrame(zip(file_names, cols, rows, heights, widths), columns = ['file', 'col', 'row', 'heights', 'width'])
df_signatures = pd.DataFrame(zip(files, target), columns = ['file', 'hasSignature'])
joblib.dump(df_bbox, 'data/signatures_bbox_df.pkl')
joblib.dump(df_bbox, 'data/signatures_df.pkl')