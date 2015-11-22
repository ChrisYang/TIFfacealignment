//Author: Heng Yang (yanghengnudt@gmail.com)
//University of Cambridge 
/*  
This code describes our method of Triplet-Interpolated Feature for landmarks 
localisation for both human and sheep faces. In the following publication:
Heng Yang*, Renqiao Zhang*, Peter Robinson, "Human and Sheep Landmarks Localisation by Triplet-Interpolated Features", WACV2016
*/

#include <opencv2/opencv.hpp>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <fstream>

#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>

using namespace dlib;
using namespace std;
using namespace cv;
// ----------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize(argv[1]) >> sp;
        std::string videoname = argv[2];
        string winname("TIF Cambridge");
        cv::namedWindow(winname, 0);
        cv::VideoCapture cap;
        if (videoname.compare("0") != 0)
            cap.open(videoname);//
        else
            cap.open(0);
        if (!cap.isOpened())
            return -1;
        int key = 0;
        std::cout << "Start human face Alignment" << std::endl;
        while (key != 27)
        {
            cv::Mat frame;
            cv::Mat gray;
            cap >> frame;
            if ( not frame .empty()) {
                array2d<unsigned char> img;
                cv::cvtColor(frame, gray, CV_BGR2GRAY);
                dlib::cv_image<unsigned char> dlibimg(gray);
                assign_image(img, dlibimg);
                std::vector<dlib::rectangle> dets = detector(img);
                cout << "Number of faces detected: " << dets.size() << endl;
                std::vector<full_object_detection> shapes;
                cv::Rect facebb;
                for (unsigned long j = 0; j < dets.size(); ++j)
                {
                    full_object_detection shape = sp(img, dets[j]);
                    cv::rectangle(frame,cv::Rect(dets[j].left(),dets[j].top(),dets[j].width(),dets[j].height()) ,cv::Scalar(128, 128, 0, 0),2);
                    for (unsigned int k = 0; k < shape.num_parts(); k++) {
                        cv::circle(frame, cv::Point_<int>(shape.part(k).x(), shape.part(k).y()), 2, cv::Scalar(0, 170, 255, 0), 2);
                    }

                }
                cv::imshow(winname, frame);
                cv::waitKey(1);
            }
        }
        cv::destroyAllWindows();
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

