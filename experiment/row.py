import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
import os

basepath ="../dicom/"
IMAGE_PATHS = os.listdir("../dicom/")

list_b = []
for f in IMAGE_PATHS:
    d = pydicom.read_file(basepath+f)
    a = np.array(d.pixel_array)
    img_2d = a.astype(float)
    img_2d_scaled = (np.maximum(img_2d,0) / img_2d.max()) * 255.0
    img_2d_scaled = np.uint8(img_2d_scaled)
    hasil = img_2d_scaled
    
    #otsu thresholding
    _,binarized = cv2.threshold(hasil, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    foreground_value = 255
    mask = np.uint8(binarized == foreground_value)
    
    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    binarized = np.zeros_like(binarized)
    binarized[labels == largest_label] = foreground_value
    
    kernel = np.ones((5, 5), np.uint8)

    #erosion
    erosion = cv2.erode(binarized,kernel,iterations = 1)
    
    foreground_value = 255
    mask = np.uint8(erosion == foreground_value)
    
    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    erosion = np.zeros_like(erosion)
    erosion[labels == largest_label] = foreground_value
    
    #opening
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    #closing
    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
    
    foreground_value = 255
    mask = np.uint8(closing == foreground_value)
    
    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    closing = np.zeros_like(closing)
    closing[labels == largest_label] = foreground_value
    
    shape = np.ones((5, 5), np.uint8)
    
    #dilasi
    dilation_open = cv2.dilate(opening,shape,iterations = 1)

    foreground_value = 255
    mask = np.uint8(dilation_open == foreground_value)

    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    dilation_open = np.zeros_like(dilation_open)
    dilation_open[labels == largest_label] = foreground_value

    #Skull Stripping
    skull_stripped_image = cv2.bitwise_and(hasil, hasil, mask = dilation_open)
    brain_pixels = skull_stripped_image[dilation_open == foreground_value]
    
    # Adapting the data to K-means
    kmeans_input = np.float32(brain_pixels.reshape(
    brain_pixels.shape[0], brain_pixels.ndim))
    
    # K-means parameters
    epsilon = 0.01
    number_of_iterations = 50
    number_of_clusters = 4
    number_of_repetition = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,number_of_iterations, epsilon)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    # K-means segmentation
    _, labels, centers = cv2.kmeans(kmeans_input, number_of_clusters, None, criteria,number_of_repetition, flags)
    
    # Adapting the labels
    labels = labels.flatten('F')
    for x in range(number_of_clusters):
        labels[labels == x] = centers[x]
    
    segmented_image = np.zeros_like(dilation_open)
    segmented_image[dilation_open == foreground_value] = labels
    
    list_b.append(segmented_image)
    
fig=plt.figure(figsize=(49, 49))
columns = 6
rows = 7
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(list_b[i])
plt.show()

