import numpy as np
import cv2, pydicom
import matplotlib.pyplot as plt

class brainSegment:
    def __init__(self, fg= 255, th= 155, it= 4, uk= 4, st_uk= 1, st_end= 3, n_it= 50, n_cl= 4, n_rp= 10):
        self.foreground = fg
        self.nilai_threshold = th
        self.iterasi = it
        self.ukuran = uk
        self.start_ukuran, self.end_ukuran = st_uk, st_end
        self.nilai_iterasi = n_it
        self.nilai_cluster = n_cl
        self.nilai_repetition = n_rp

    def bukadata(self, file= 'Z102'):    
        # get the data
        d = pydicom.read_file('dicom/'+file)
        file = np.array(d.pixel_array)
        img = self.file
        img_2d = img.astype(float)
        img_2d_scaled = (np.maximum(img_2d,0) / img_2d.max()) * self.foreground
        img_2d_scaled = np.uint8(img_2d_scaled)
        hasil = img_2d_scaled
        plt.imshow(hasil, caption='Gambar Origin', use_column_width=True)
        return hasil

    def otsuthreshold(self, image):
        #OTSU THRESHOLDING
        _,binarized = cv2.threshold(image, 0, self.foreground, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        foreground_value = self.foreground
        mask = np.uint8(binarized == foreground_value)
        labels, stats = cv2.connectedComponentsWithStats(mask, self.ukuran)[self.start_ukuran:self.end_ukuran]
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binarized = np.zeros_like(binarized)
        binarized[labels == largest_label] = foreground_value
        plt.imshow(binarized, caption='Otsu Image', use_column_width=True)
        return binarized

    def gaussianthreshold(self, image):
        gaussian = cv2.adaptiveThreshold(image, self.foreground, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,self.nilai_threshold, 1)
        # masking(gaussian)
        foreground_value = self.foreground
        mask = np.uint8(gaussian == foreground_value)
        labels, stats = cv2.connectedComponentsWithStats(mask, self.ukuran)[self.start_ukuran:self.end_ukuran]
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        gaussian = np.zeros_like(gaussian)
        gaussian[labels == largest_label] = foreground_value
        plt.imshow(gaussian, caption='Gaussian Image', use_column_width=True)
        return gaussian

    def erosion(self, image):
        # erosion from otsu
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.end_ukuran, self.end_ukuran))
        erosion = cv2.erode(image,kernel,iterations = self.iterasi)
        foreground_value = self.foreground
        mask = np.uint8(erosion == foreground_value)
        labels, stats = cv2.connectedComponentsWithStats(mask, self.ukuran)[self.start_ukuran:self.end_ukuran]
        largest_label = 1 + np.argmax(stats[self.start_ukuran:, cv2.CC_STAT_AREA])
        erosion = np.zeros_like(erosion)
        erosion[labels == largest_label] = foreground_value
        plt.imshow(erosion, caption='Erosion Image', use_column_width=True)
        return erosion

    def opening(self,image):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.end_ukuran,self.end_ukuran))
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations= self.iterasi)
        foreground_value = self.foreground
        mask = np.uint8(opening == foreground_value)
        labels, stats = cv2.connectedComponentsWithStats(mask, self.ukuran)[self.start_ukuran: self.end_ukuran]
        largest_label = 1 + np.argmax(stats[self.start_ukuran:, cv2.CC_STAT_AREA])
        opening = np.zeros_like(opening)
        opening[labels == largest_label] = foreground_value
        plt.imshow(opening, caption='Opening Image', use_column_width=True)
        return opening

    def closing(self, image):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.end_ukuran, self.end_ukuran))
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations= self.iterasi)
        foreground_value = self.foreground
        mask_closing = np.uint8(closing >= foreground_value)
        labels, stats = cv2.connectedComponentsWithStats(mask_closing, self.ukuran)[self.start_ukuran:self.end_ukuran]
        largest_label = 1 + np.argmax(stats[self.start_ukuran:, cv2.CC_STAT_AREA])
        close = np.zeros_like(closing)
        close[labels == largest_label] = foreground_value
        plt.imshow(close, caption='Closing Image', use_column_width=True)
        return close

    def dilation(self, image):
        # dilation from opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.end_ukuran, self.end_ukuran))
        dilasi = cv2.dilate(image,kernel,iterations = self.iterasi)
        foreground_value = self.foreground
        mask = np.uint8(dilasi == foreground_value)
        labels, stats = cv2.connectedComponentsWithStats(mask, self.ukuran)[self.start_ukuran: self.end_ukuran]
        largest_label = 1 + np.argmax(stats[self.start_ukuran:, cv2.CC_STAT_AREA])
        dilasi = np.zeros_like(dilasi)
        dilasi[labels == largest_label] = foreground_value
        plt.imshow(dilasi, caption='Dilation Image', use_column_width=True)
        return dilasi

    def cluster(self, image, dilasi, foreground_value):
        #Skull Stripping
        skull_stripped_image = cv2.bitwise_and(image, image, mask = dilasi)
        brain_pixels = skull_stripped_image[dilasi == foreground_value]

        # Adapting the data to K-means
        kmeans_input = np.float32(brain_pixels.reshape(brain_pixels.shape[0], brain_pixels.ndim))
        
        # K-means parameters
        epsilon = 0.01
        number_of_iterations = self.nilai_iterasi
        number_of_clusters = self.nilai_cluster
        number_of_repetition = self.nilai_repetition
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
        plt.imshow(segmented_image, caption='Segmented Image', use_column_width=True)
        return segmented_image

    def divided(self,image, a=0, b=0, c=0, jml_a=0, jml_b=0, jml_c=0, jml_d=0):  
        segmented_image = image    
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
                segmented_image[segmented_image!=a] = 0
            elif jml_b>jml_a and jml_b>jml_c:
                segmented_image[segmented_image!=b]=0
            elif jml_c>jml_a and jml_c>jml_b:
                segmented_image[segmented_image!=c]=0

        plt.imshow(segmented_image, caption='Divided Image', use_column_width=True)
        return segmented_image

if __name__ == "__main__":
    foreground = 255
    mySegmen = brainSegment()
    a = mySegmen.bukadata()
    b = mySegmen.gaussianthreshold(a)
    c = mySegmen.erosion(b)
    d = mySegmen.closing(c)
    e = mySegmen.dilation(d)
    f = mySegmen.cluster(a, e, foreground)