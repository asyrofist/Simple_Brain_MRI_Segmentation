import numpy as np
import cv2

# Image loading
image = cv2.imread('ImageT2.png', 0)

# Filtering
filtered = cv2.medianBlur(image, 3)

# Otsu thresholding
_, binarized = cv2.threshold(
    filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Largest connected component (llc)
foreground_value = 255
mask = np.uint8(binarized == foreground_value)
labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
llc = np.zeros_like(binarized)
llc[labels == largest_label] = foreground_value

# Mathematical morphology
kernel = np.ones((5, 5), np.uint8)
llc_closing_image = cv2.morphologyEx(llc, cv2.MORPH_CLOSE, kernel)

# Skull stripping
skull_stripped_image = cv2.bitwise_and(
    filtered, filtered, mask=llc_closing_image)
brain_pixels = skull_stripped_image[llc_closing_image == foreground_value]

# Adapting the data to K-means
kmeans_input = np.float32(brain_pixels.reshape(
    brain_pixels.shape[0], brain_pixels.ndim))

# K-means parameters
epsilon = 0.01
number_of_iterations = 50
number_of_clusters = 4
number_of_repetition = 10
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            number_of_iterations, epsilon)
flags = cv2.KMEANS_RANDOM_CENTERS

# K-means segmentation
_, labels, centers = cv2.kmeans(
    kmeans_input, number_of_clusters, None, criteria,
    number_of_repetition, flags)

# Adapting the labels
labels = labels.flatten('F')
for x in range(number_of_clusters):
    labels[labels == x] = centers[x]

# Segmented image
segmented_image = np.zeros_like(llc_closing_image)
segmented_image[llc_closing_image == foreground_value] = labels

# Display
captions = ["1. Original image", "2. Thresholding",
            "3. Largest connected component + Mathematical morphology",
            "4. Segmented image"]
horizontal_layout = np.hstack(
    (image, binarized, llc_closing_image, segmented_image))
cv2.imshow(" | ".join(captions), horizontal_layout)

cv2.waitKey(0)
