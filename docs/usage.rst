============
Instalasi
============

Install langsung dari pypi
-------------------

Install the package with pip::

    $ pip install brain_segmen

Cara ini digunakan untuk pengguna dapat menggunakan library ini. adapun untuk menginstal dapat menggunakan cara lain, seperti langsung menginstal dari repository package library ini dari 

Install Langsung dari Repository
-------------------

Atau jika terlalu susah untuk menginstalnya, tinggal download berkas dari laman berikut ini. Cukup mudah dan simpel tinggal klik bagian code dan download zip tersebut.

**Download extract-req**: https://github.com/asyrofist/brain_segmen

::

    cd brain_segmen
    python setup.py install
    # If root permission is not available, add --user command like below
    # python setup.py install --user

Currently, pip install is not supported but will be addressed.


Instalasi Keterlacakan Dokumen SKPL
------------------------------------------
Library ini dapat digunakan menggunakan spesifikasi dibawah ini, dimana python dan requirement yang dibutuhkan adalah sebagai berikut.
karena pengembangan menggunakan environment 3.7 maka disarankan untuk menginstal diatas versi tersebut.

- Python :code:`>= 3.7`

Requirements
------------
Dalam instalasi ini, membutuhkan package yang lain seperti daftar berikut ini. anda bisa melihatnya di 
(bagian depan repository github saya yang berada di :doc:`/requirement.txt` section.) 
Segala macam detail saya jelaskan pada sebagai berikut.

- numpy
- opencv-python-headless
- pydicom
- matplotlib
- scikit-image
- Pillow


========
Penggunaan
========
Contoh Penggunaan Library
------------

Bagaimana cara menggunakan template ini, dapat dilihat dari contoh syntax berikut ini::

  from brainsegmen.segmen import brainSegment
  mySegmen = brainSegment()
  a = mySegmen.bukadata(data)

Berikut ini penjelasan singkat darri contoh syntax tersebut.

- mySegmen.bukadata(data)
Bagian ini untuk membuka data dicom yang diuji sebagai bagian analisis. dari percobaan ini, nantinya pengembangang dapat melihat hasil percobaan tersebut.
