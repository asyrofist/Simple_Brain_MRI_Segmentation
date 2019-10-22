import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
import os

# Function
def ShowImage(title,img,ctype):
  plt.figure(figsize=(9, 9))
  if ctype=='gray':
    plt.imshow(img,cmap='gray')
  elif ctype=='rgb':
    plt.imshow(img)
  else:
    raise Exception("Unknown colour type")
  plt.axis('off')
  plt.title(title)
  plt.show()

def masking(image):
    foreground_value = 255
    mask = np.uint8(image == foreground_value)

    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    image = np.zeros_like(image)
    image[labels == largest_label] = foreground_value

basepath ="../dicom/"
d = pydicom.read_file(basepath + "Z108")
file = np.array(d.pixel_array)  
img = file

img_2d = img.astype(float)
img_2d_scaled = (np.maximum(img_2d,0) / img_2d.max()) * 255.0
img_2d_scaled = np.uint8(img_2d_scaled)
hasil = img_2d_scaled

name = input('masukan kondisi: ')


if (name == '1'):
  
  #OTSU THRESHOLDING
  _,binarized = cv2.threshold(hasil, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  foreground_value = 255
  mask = np.uint8(binarized == foreground_value)

  labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
  largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
  binarized = np.zeros_like(binarized)
  binarized[labels == largest_label] = foreground_value

  # erosion
  kernel = np.ones((5, 5), np.uint8)
  erosion = cv2.erode(binarized,kernel,iterations = 1)
  masking(erosion)

  bentuk = np.ones((5, 5), np.uint8)
  opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, bentuk)
  masking(opening)

  # closing
  kernel = np.ones((5, 5), np.uint8)
  closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
  masking(closing)

  # dilasi
  shape = np.ones((5, 5), np.uint8)
  dilation_open = cv2.dilate(opening,shape,iterations = 1)

  foreground_value = 255
  mask = np.uint8(dilation_open == foreground_value)

  labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
  largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
  dilation_open = np.zeros_like(dilation_open)
  dilation_open[labels == largest_label] = foreground_value

  img_2d = file.astype(float)
  img_2d_scaled = (np.maximum(img_2d,0) / img_2d.max()) * 255.0
  img_2d_scaled = np.uint8(img_2d_scaled)
  hasil = img_2d_scaled

  #Skull Stripping
  skull_stripped_image = cv2.bitwise_and(hasil, hasil, mask = dilation_open)
  brain_pixels = skull_stripped_image[dilation_open == foreground_value]

  # Adapting the data to K-means
  kmeans_input = np.float32(brain_pixels.reshape(brain_pixels.shape[0], brain_pixels.ndim))

  # K-means parameters
  epsilon = 0.01
  number_of_iterations = 50
  number_of_clusters = 4
  number_of_repetition = 10
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, number_of_iterations, epsilon)
  flags = cv2.KMEANS_RANDOM_CENTERS

  # K-means segmentation
  _, labels, centers = cv2.kmeans(kmeans_input, number_of_clusters, None, criteria, number_of_repetition, flags)

  # Adapting the labels
  labels = labels.flatten('F')
  for x in range(number_of_clusters):
      labels[labels == x] = centers[x]

  # segmentasiclea
  segmented_image = np.zeros_like(dilation_open)
  segmented_image[dilation_open == foreground_value] = labels


  # Display
  captions = ["1. Original image", "2. Thresholding + Masking", "3. Morphology", "4. Segmented image"]
  horizontal_layout = np.hstack((hasil, binarized, dilation_open, segmented_image))
  cv2.imshow(" | ".join(captions), horizontal_layout)
  cv2.waitKey(0)

if (name == '2'):
  gaussian = cv2.adaptiveThreshold(hasil,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,115, 1)
  foreground_value = 255
  mask = np.uint8(gaussian == foreground_value)

  labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
  largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
  gaussian = np.zeros_like(gaussian)
  gaussian[labels == largest_label] = foreground_value
  
  kernel = np.ones((5, 5), np.uint8)
  erosion = cv2.erode(gaussian,kernel,iterations = 1)
  
  foreground_value = 255
  mask_erosion = np.uint8(erosion == foreground_value)

  labels, stats = cv2.connectedComponentsWithStats(mask_erosion, 4)[1:3]
  largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
  erosion = np.zeros_like(erosion)
  erosion[labels == largest_label] = foreground_value

  closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)

  foreground_value = 255
  mask_closing = np.uint8(closing >= foreground_value)

  labels, stats = cv2.connectedComponentsWithStats(mask_closing, 4)[1:3]
  largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
  close = np.zeros_like(closing)
  close[labels == largest_label] = foreground_value
  
  bentukan = np.ones((5, 5), np.uint8)
  dilasi = cv2.dilate(closing,bentukan,iterations = 1)

  foreground_value = 255
  mask_dilasi = np.uint8(dilasi >= foreground_value)

  labels, stats = cv2.connectedComponentsWithStats(mask_dilasi, 4)[1:3]
  largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
  tutup = np.zeros_like(dilasi)
  tutup[labels == largest_label] = foreground_value


  #Skull Stripping
  skull_stripped_image = cv2.bitwise_and(hasil, hasil, mask = tutup)
  brain_pixels = skull_stripped_image[tutup == foreground_value]

  # Adapting the data to K-means
  kmeans_input = np.float32(brain_pixels.reshape(brain_pixels.shape[0], brain_pixels.ndim))

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

  segmented_image = np.zeros_like(tutup)
  segmented_image[tutup == foreground_value] = labels
  
  # Display
  captions = ["1. Original image", "2. Thresholding + Masking", "3. Morphology", "4. Segmented image"]
  horizontal_layout = np.hstack((hasil, gaussian, tutup, segmented_image))
  cv2.imshow(" | ".join(captions), horizontal_layout)
  cv2.waitKey(0)