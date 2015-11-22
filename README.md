# TIFfacealignment
This code is used for human and sheep face alignment using our proposed Triplet-Interpolated-Feature method. 

![alt tag](https://github.com/ChrisYang/TIFfacealignment/blob/master/sheep.jpg)

[Heng Yang](https://sites.google.com/site/yanghengcv/home), Renqiao Zhang*, Peter Robinson, ["Human and Sheep Landmarks Localisation by Triplet-Interpolated Features"](http://arxiv.org/pdf/1509.04954.pdf), WACV2016 (First two authors contribute equally.)

The work was mainly done when Renqiao Zhang was an intern working with me at Rainbow Group, Computer Laboratory, University of Cambridge. The sheep data were collected by him either from Internet or from Dr. Krista McLennan at Animal Welfare and Anthrozoology at University of Cambridge. If you use the code or data please cite the above paper. 

Please note that our main code is put inside the dlib folder called:

*./dlib-18.16/dlib/image_processing/shape_predictor_TIF.h*

We keep the format consistent with the original shape_predictor.h


Currently tested on Mac OS. If you are using OpenCV3.0 or above please change the library names.

For **sheep facial landmarks localisation**: 

* Specify the opencv and boost path in Makefile 
* $ make -f Makefile_sheep
* $ ./TIF_sheep Model/sheep_8p.dat imagelist.txt 

Then you will see the localisation results in each image. If you want to speed it up, change cv::waitKey(1000) to cv::waitKey(1) in facealignment.cpp. 
After running the code, the result will be saved to *filename*_result.txt 

For **human facial landmarks localisation (face alignment)**:

* Specify the opencv and boost path in Makefile 
* $ make -f Makefile_human
* $ ./TIF_human Model/TIF_face.dat *videoname* 

If videoname is *0*, it opens a camera. Otherwise it will open videoname.  It then detects faces in each frame using the face detector from dlib and applies face alignment on each detected face. 





