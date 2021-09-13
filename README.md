# Brain Segmentation
Author  | [Rakha Asyrofi](https://scholar.google.com/citations?user=WN9T5UUAAAAJ&hl=id&oi=ao)
 -------|-----------
Version | 0.0.1
Updated | 2021-09-12

# Overview
Click this link to see [live demo](https://segbrain.herokuapp.com/) If you want to see 

Cerebellum and Frontal Lobe Segmentation Based on K-Means Clustering and Morphological Transformation described in [our paper at iSemantic2020](https://ieeexplore.ieee.org/document/9234262). Please kindly cite the following paper when you use this tool. It would also be appreciated if you send me a courtesy [website](http://rakha.asyrofi.com/) and [google scholar](https://scholar.google.com/citations?user=WN9T5UUAAAAJ&hl=id&oi=ao), so I could survey what kind of tasks the tool is used for. 
```
@inproceedings{rakha-asyrofi-2020-brainsegmentation,
    title = "Cerebellum and Frontal Lobe Segmentation Based on K-Means Clustering and Morphological Transformation",
    author = "Asyrofi, Rakha  and
      Winata, Yoni Azhar  and
      Sarno, Riyanarto  and
      Fajar, Aziz",
    booktitle = "2020 International Seminar on Application for Technology of Information and Communication (iSemantic)",
    month = October,
    year = "2020",
    address = "Semarang, Indonesia",
    publisher = "IEEE",
    url = "https://doi.org/10.1109/iSemantic50169.2020.9234262",
    pages = "149-154",
    language = "English",
    ISBN = "9781728190686",
}
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



