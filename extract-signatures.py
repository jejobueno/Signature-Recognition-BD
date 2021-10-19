import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
import numpy as np

# the parameters are used to remove small size connected pixels outliar
constant_parameter_1 = 84
constant_parameter_2 = 250
constant_parameter_3 = 350

# the parameter is used to remove big size connected pixels outliar
constant_parameter_4 = 260


img = cv2.imread('data/train/0d178d095434170eac2cb58cc244bb8c_2.tif')
# read the input image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kernel = np.ones((7,7), np.uint8)
img_erosion = cv2.erode(gray, kernel, iterations=1)
#img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
img = cv2.threshold(img_erosion, 160, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Remove horizontal
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

# Repair image
repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
result = 255 - cv2.morphologyEx(255 - img, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
#img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
plt.imshow(img)
plt.show()

# connected component analysis by scikit-learn framework
blobs = img > img.mean()
blobs_labels = measure.label(blobs, background=1)
image_label_overlay = label2rgb(blobs_labels, image=img)

fig, ax = plt.subplots(figsize=(10, 6))


# plot the connected components (for debugging)
ax.imshow(image_label_overlay)
ax.set_axis_off()
plt.title('Labels')
plt.tight_layout()
plt.show()

the_biggest_component = 0
total_area = 0
counter = 0
average = 0.0
areas = []
for region in regionprops(blobs_labels):
    areas.append(region.area)
    if (region.area > 10):
        total_area = total_area + region.area
        counter = counter + 1

    # print region.area # (for debugging)
    # take regions with large enough areas
    if region.area >= 18000:
        if region.area > the_biggest_component:
            [x, y, w, h] = region.bbox
            points = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                              dtype=np.int32)
            # draw rectangle around contour on original image
            # cv2.fillPoly(img, [points], (255, 255, 255))
            the_biggest_component = region.area


average = (total_area / counter)
print('Components found:' + str(counter))
print("the_biggest_component: " + str(the_biggest_component))
print("average: " + str(average))

# experimental-based ratio calculation, modify it for your cases
# a4_small_size_outliar_constant is used as a threshold value to remove connected outliar connected pixels
# are smaller than a4_small_size_outliar_constant for A4 size scanned documents
a4_small_size_outliar_constant = ((average / constant_parameter_1) * the_biggest_component) + constant_parameter_3
print("a4_small_size_outliar_constant: " + str(a4_small_size_outliar_constant))

# experimental-based ratio calculation, modify it for your cases
# a4_big_size_outliar_constant is used as a threshold value to remove outliar connected pixels
# are bigger than a4_big_size_outliar_constant for A4 size scanned documents
a4_big_size_outliar_constant = a4_small_size_outliar_constant * constant_parameter_4
print("a4_big_size_outliar_constant: " + str(a4_big_size_outliar_constant))

# remove the connected pixels are smaller than a4_small_size_outliar_constant
pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)
# remove the connected pixels are bigger than threshold a4_big_size_outliar_constant
# to get rid of undesired connected pixels such as table headers and etc.
component_sizes = np.bincount(pre_version.ravel())
too_small = component_sizes > (a4_big_size_outliar_constant)
too_small_mask = too_small[pre_version]
pre_version[too_small_mask] = 0
# save the the pre-version which is the image is labelled with colors
# as considering connected components
plt.imsave('pre_version.png', pre_version)

# read the pre-version
img = cv2.imread('pre_version.png', 0)
# ensure binary
img = cv2.threshold(img, 0, 255,
                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# save the the result
plt.figure()
plt.imshow(img)
plt.show()




