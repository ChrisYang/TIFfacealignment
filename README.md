# TIFfacealignment
This code is used for human and sheep face alignment using our proposed Triplet-Interpolated-Feature method. 

[Heng Yang](https://sites.google.com/site/yanghengcv/home), Renqiao Zhang*, Peter Robinson, ["Human and Sheep Landmarks Localisation by Triplet-Interpolated Features"](http://arxiv.org/pdf/1509.04954.pdf), WACV2016 (First two authors made equal contribution.)

The work was mainly done when Renqiao Zhang was an intern working with me at Rainbow Group, Computer Laboratory, University of Cambridge. The sheep data were collected by him either from Internet or from Dr. Krista McLennan at Animal Welfare and Anthrozoology at University of Cambridge. If you use the code or data please cite the above paper. 

To run to code. (Currently tested on Mac OS. If you are using OpenCV3.0 or above please change the library names.) 

* Specify the opencv and boost path in Makefile 
* $ make
* $ ./TIF Model/sheep_8p.dat imagelist.txt 

Then you will see the localisation results in each image. If you want to speed it up, change cv::waitKey(1000) to cv::waitKey(1) in facealignment.cpp. Please note that our main code is put inside the dlib folder called:

**./dlib-18.16/dlib/image_processing/shape_predictor_TIF.h**

We keep the format consistent with the original shape_predictor.h

After running the code, the result will be saved to *filename*_result.txt 






