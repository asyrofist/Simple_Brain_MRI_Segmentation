![PyPI - Python Version](https://img.shields.io/badge/python-3.7.0-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/brain-segmen.svg)](https://pypi.org/project/brain-segmen/)
![image](https://visitor-badge.laobi.icu/badge?page_id=asyrofist/Brain_segmentation)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
[![Build Status](https://app.travis-ci.com/asyrofist/Brain-Segmentation.svg?branch=master)](https://app.travis-ci.com/asyrofist/Brain-Segmentation)
[![Paper](http://img.shields.io/badge/Paper-PDF-red.svg)](https://ieeexplore.ieee.org/document/9234262)

# Brain Segmentation
Author  | [Rakha Asyrofi](https://scholar.google.com/citations?user=WN9T5UUAAAAJ&hl=id&oi=ao)
 -------|-----------
Version | 0.0.1
Updated | 2021-09-12

# Overview
Click this link to see [live demo](https://segbrain.herokuapp.com/) If you want to see 

Cerebellum and Frontal Lobe Segmentation Based on K-Means Clustering and Morphological Transformation described in [our paper at iSemantic2020](https://ieeexplore.ieee.org/document/9234262). Please kindly cite the following paper when you use this tool. It would also be appreciated if you send me a courtesy [website](http://rakha.asyrofi.com/) and [google scholar](https://scholar.google.com/citations?user=WN9T5UUAAAAJ&hl=id&oi=ao), so I could survey what kind of tasks the tool is used for. 
```
@INPROCEEDINGS{9234262,  author={Asyrofi, Rakha and Winata, Yoni Azhar and Sarno, Riyanarto and Fajar, Aziz},  
booktitle={2020 International Seminar on Application for Technology of Information and Communication (iSemantic)},   
title={Cerebellum and Frontal Lobe Segmentation Based on K-Means Clustering and Morphological Transformation},   
year={2020},  
volume={},  
number={},  
pages={149-154},  
doi={10.1109/iSemantic50169.2020.9234262}}
```

Developed by Asyrofi (c) 2021

## Cara menginstal

instalasi melalui pypi:

    pip install brain_segmen


## Cara menggunakan program

```python
from brainsegmen.segmen import brainSegment
mySegmen = brainSegment()
a = mySegmen.bukadata(data)
```

Check out: https://youtu.be/ZzlAgxbpK4E



