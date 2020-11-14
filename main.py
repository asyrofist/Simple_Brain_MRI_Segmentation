import streamlit as st
import numpy as np
import os 
import cv2
import pydicom
import matplotlib.pyplot as plt
import skimage.segmentation as seg
from PIL import Image

st.write("""
# Brain Segmenntation
Berikut ini algoritma yang digunakan untuk Segmentasi Otak
""")

IMAGE_PATHS = os.listdir("dicom")
option = st.sidebar.selectbox('Pilih File Dicom?',IMAGE_PATHS)
st.sidebar.write('You selected:', option)

def bukadata(file):    
    # get the data
    d = pydicom.read_file('dicom/'+file)
    file = np.array(d.pixel_array)
    img = file
    img_2d = img.astype(float)
    img_2d_scaled = (np.maximum(img_2d,0) / img_2d.max()) * 255.0
    img_2d_scaled = np.uint8(img_2d_scaled)
    hasil = img_2d_scaled
    st.image(hasil, caption='Gambar Origin')
    return hasil

def threshold(image):
    #OTSU THRESHOLDING
    _,binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground_value = 255
    mask = np.uint8(binarized == foreground_value)
    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    binarized = np.zeros_like(binarized)
    binarized[labels == largest_label] = foreground_value
    st.image(binarized, caption='Otsu Image')
    return binarized

def erosion(image):
    # erosion from otsu
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erosion = cv2.erode(image,kernel,iterations = 4)
    foreground_value = 255
    mask = np.uint8(erosion == foreground_value)
    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    erosion = np.zeros_like(erosion)
    erosion[labels == largest_label] = foreground_value
    st.image(erosion, caption='Erosion Image')
    return erosion

def dilation(image):
    # dilation from opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dilasi = cv2.dilate(image,kernel,iterations = 2)
    foreground_value = 255
    mask = np.uint8(dilasi == foreground_value)
    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    dilasi = np.zeros_like(dilasi)
    dilasi[labels == largest_label] = foreground_value
    st.image(dilasi, caption='Dilation Image')
    return dilasi

def cluster(image, dilasi):
    #Skull Stripping
    skull_stripped_image = cv2.bitwise_and(image, image, mask = dilasi)
    brain_pixels = skull_stripped_image[dilasi == foreground_value]

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
    #segmented Image
    segmented_image = np.zeros_like(dilasi)
    segmented_image[dilasi == foreground_value] = labels
    st.image(segmented_image, caption='Segmented Image')
    return segmented_image

a = bukadata(option)
b = threshold(a)
c = erosion(b)
d = dilation(c)
cluster(a,d)
