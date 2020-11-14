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

def otsuthreshold(image):
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

def gaussianthreshold(image):
    gaussian = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,115, 1)
    # masking(gaussian)
    foreground_value = 255
    mask = np.uint8(gaussian == foreground_value)
    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    gaussian = np.zeros_like(gaussian)
    gaussian[labels == largest_label] = foreground_value
    st.image(gaussian, caption='Gaussian Image')
    return gaussian
    
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

def opening(image):
    # kernel = np.ones((5, 5), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations= 2)
    foreground_value = 255
    mask = np.uint8(opening == foreground_value)
    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    opening = np.zeros_like(opening)
    opening[labels == largest_label] = foreground_value
    st.image(opening, caption='Opening Image')
    return opening

def closing(image):
    kernel = np.ones((5, 5), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations= 2)
    foreground_value = 255
    mask_closing = np.uint8(closing >= foreground_value)
    labels, stats = cv2.connectedComponentsWithStats(mask_closing, 4)[1:3]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    close = np.zeros_like(closing)
    close[labels == largest_label] = foreground_value
    st.image(close, caption='Closing Image')
    return close

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

def cluster(image, dilasi, foreground_value):
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

def divided(image, a=0, b=0, c=0, jml_a=0, jml_b=0, jml_c=0, jml_d=0):  
    segmented_image = image
    hasil_image = segmented_image
    
    for x in range(256):
        for y in range(256):
            if segmented_image[x][y] == 0:
                jml_d = jml_d + 1
            elif segmented_image[x][y] != 0 and a == 0:
                a = segmented_image[x][y]
                jml_a = jml_a + 1
            elif segmented_image[x][y]!=0 and segmented_image[x][y]!=a and b==0:
                b=segmented_image[x][y]
                jml_b=jml_b+1
            elif segmented_image[x][y]!=0 and segmented_image[x][y]!=a and segmented_image[x][y]!=b and c==0:
                c=segmented_image[x][y]
                jml_c=jml_c+1
            elif segmented_image[x][y] == a:
                jml_a = jml_a + 1
            elif segmented_image[x][y]==b:
                jml_b=jml_b+1
            elif segmented_image[x][y]==c:
                jml_c=jml_c+1
                
    if jml_a>jml_b and jml_a>jml_c and jml_a>jml_d:
        if jml_b>jml_c and jml_b>jml_d:
            segmented_image[segmented_image!=b]=0
        elif jml_c>jml_b and jml_c>jml_d:
            segmented_image[segmented_image!=c]=0
        elif jml_d>jml_b and jml_d>jml_c:
            segmented_image[segmented_image!=d]=0
    elif jml_b>jml_a and jml_b>jml_c and jml_b>jml_d:
        if jml_a>jml_c and jml_a>jml_d:
            segmented_image[segmented_image!=a]=0
        elif jml_c>jml_a and jml_c>jml_d:
            segmented_image[segmented_image!=c]=0
        elif jml_d>jml_a and jml_d>jml_c:
            segmented_image[segmented_image!=d]=0
    elif jml_c>jml_a and jml_c>jml_b and jml_c>jml_d:
        if jml_a>jml_b and jml_a>jml_d:
            segmented_image[segmented_image!=a]=0
        elif jml_b>jml_a and jml_b>jml_d:
            segmented_image[segmented_image!=b]=0
        elif jml_d>jml_a and jml_d>jml_b:
            segmented_image[segmented_image!=d]=0
    elif jml_d>jml_a and jml_d>jml_b and jml_d>jml_c:
        if jml_a>jml_b and jml_a>jml_c:
            hasil_image[segmented_image!=a] = 0
        elif jml_b>jml_a and jml_b>jml_c:
            segmented_image[segmented_image!=b]=0
        elif jml_c>jml_a and jml_c>jml_b:
            segmented_image[segmented_image!=c]=0

    st.image(hasil_image, caption='Divided Image')
    return hasil_image


morphology1a = st.sidebar.checkbox('Morphology1 (Otsu-Erosion-Dilation-cluster)')
morphology2a = st.sidebar.checkbox('Morphology2 (Gaussian-Erosion-Dilation-cluster)')
morphology1b = st.sidebar.checkbox('Improvement Morphology1 (Otsu-Erosion-Opening-Dilation-cluster)')
morphology2b = st.sidebar.checkbox('Improvement Morphology2 (Gaussian-Erosion-Closing-Dilation-cluster)')
morphology3 = st.sidebar.checkbox('Morphology3 (Otsu-cluster-Erosion-Dilation)')

if morphology1a:
    a = bukadata(option)
    b = otsuthreshold(a)
    c = erosion(b)
    d = dilation(c)
    cluster(a, d, 255)

elif morphology2a:
    a = bukadata(option)
    b = gaussianthreshold(a)
    c = erosion(b)
    d = dilation(c)
    cluster(a, d, 255)
    
elif morphology1b:
    a = bukadata(option)
    b = otsuthreshold(a)
    c = erosion(b)
    d = opening(c)
    e = dilation(d)
    cluster(a, e, 255)
    
elif morphology2b:
    a = bukadata(option)
    b = gaussianthreshold(a)
    c = erosion(b)
    d = closing(c)
    e = dilation(d)
    cluster(a, e, 255)
    
elif morphology2b:
    a = bukadata(option)
    b = gaussianthreshold(a)
    c = erosion(b)
    d = closing(c)
    e = dilation(d)
    cluster(a, e, 255)

elif morphology3:
    a = bukadata(option)
    b = otsuthreshold(a)
    c = cluster(a, b, 255)
#     d = divided(c)
    erosion(c)
#     f = closing(e)
#     dilation(f)

    
