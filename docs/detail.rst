Cerebellum and Frontal Lobe Segmentation Based on K-Means Clustering and Morphological Transformation
=========================================================================================

Overview
------------

Secara garis besar, library ini dibuat untuk mengekstraksi cerebelum, dan frontal lobe segementation berdasarkan K-Means Clustering dan Morphological Transformation, 
yang telah diterangkan pada proceeding conference di  `Isemantic2020`_. 
Jika anda menggunakan library ini, saya sangat mengapresiasi, dengan cara mengirimkan segala macam bentuk kiriman melalui `courtesy`_  dan `scholar`_, 
Semoga data yang saya publikasikan, berguna untuk orang banyak, terima kasih. 

Abstrak
------------
K-means clustering can be used as an algorithm segmentation that can split an area of interest from the image into several different regions containing each pixel based on color. 
Nevertheless, the result of the color division of the clustering has not been able to display clean segmentation because there are still pixels that connect each other and produce pixel noise or unwanted pixels. 
In this work, we propose a technique where it can select four dominant colors from the k-means clustering results then display it as digital image output. In our approach, the proposed method can separate the cerebellum and frontal lobe from the background of the brain after several operations of morphological transformation. 
In implementing this method, three samples of the brain from different people were tested. From the experimental results, the DSI produces a value of 0.72 from 1 for the frontal lobe and 0.86 from 1 for the cerebellum. It means that the proposed method can segment the desired part of the brain properly.

.. _Isemantic2020: https://ieeexplore.ieee.org/document/9234262
.. _courtesy: https://www.researchgate.net/profile/Rakha_Asyrofi
.. _scholar: https://scholar.google.com/citations?user=WN9T5UUAAAAJ&hl=id&oi=ao

Dikembangkan oleh Rakha Asyrofi (c) 2021

Cara menginstall
--------------

Instalasi menggunakan PYPI:

    pip install brain_segmen

Fitur yang digunakan
------------
Berikut ini adalah beberapa fitur yang telah digunakan sebagai berikut:

- library ini dapat mengekstraksi bagian otak yaitu frontal lobe dan cerebelum

Kontribusi
------------
Sebagai bahan pengemabangan saya, maka saya apresiasi apabila anda, dapat mengecek issue dari repository library ini.

- Issue Tracker: https://github.com/asyrofist/brain_segmen/issues
- Source Code: https://github.com/asyrofist/brain_segmen

Support
------------
Jika anda memiliki masalah, saat menggunakan library ini. Mohon dapat membuat mailing list ke at: asyrofi.19051@mhs.its.ac.id

Lisensi
------------
Proyek ini memiliki lisensi atas MIT License
