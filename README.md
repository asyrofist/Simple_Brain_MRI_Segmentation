# Simple brain MRI Segmentation
This script shows how to process an axial brain MRI image to perform skull stripping and segmentation of the brain.

The skull stripping is performed through:

* Otsu thresholding.
* Computation of the largest connected component.
* Mathematical morphology.

Once the brain extracted, it is given to the K-Means clustering algorithm to peform the segmentation.

Note that the image contains white matter, gray matter, cerebrospinal fluid as well as multiple sclerosis lesions.
