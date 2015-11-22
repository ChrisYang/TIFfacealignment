//Author: Heng Yang (yanghengnudt@gmail.com)
//University of Cambridge 
/*  
This code describes our method of Triplet-Interpolated Feature for landmarks 
localisation for both human and sheep faces. In the following publication:
Heng Yang*, Renqiao Zhang*, Peter Robinson, "Human and Sheep Landmarks Localisation by Triplet-Interpolated Features", WACV2016

*/


#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <iostream>
#include <fstream>
#include <time.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/matrix.h>
#include <ctime>
using namespace dlib;
using namespace cv;
using namespace std;

// ----------------------------------------------------------------------------------------
bool load_annotations(std::vector<std::string>& names, std::vector<cv::Rect>& rects, std::string url) {
  if (boost::filesystem::exists(url.c_str())) {
    std::string filename(url.c_str());
    boost::iostreams::stream < boost::iostreams::file_source > file(
        filename.c_str());
    std::string line;
    names.clear();
    rects.clear();
    while (std::getline(file, line)) {
      std::vector < std::string > strs;
      boost::split(strs, line, boost::is_any_of(" "));
      names.push_back(strs[0]);
      cv::Rect rect;
      rect.x = std::atoi(strs[1].c_str());
      rect.y = std::atoi(strs[2].c_str());
      rect.width = std::atoi(strs[3].c_str());
      rect.height = std::atoi(strs[4].c_str());
      rects.push_back(rect);
    }
    return true;
  }
  return false;
}
int main(int argc, char** argv)
{  
    try
    {

        if (argc == 1)
        {
            cout << "Call this program like this:" << endl;
            cout << "./TIF Model/sheep_8p.dat images.txt" << endl;
            return 0;
        }

        shape_predictor sp;
        deserialize(argv[1]) >> sp;
        cout << "This program detects " << sp.num_parts() << " landmarks" << endl;
        std::string imgsfilename = argv[2];
        std::string outfilename = imgsfilename;
        outfilename.replace(outfilename.end()-4, outfilename.end(),"_TIF_result.txt");
        ofstream out(outfilename.c_str());
        std::cout << "start processing..." << std::endl;
        std::vector<std::string> names;
         std::vector<cv::Rect> rects;
        load_annotations(names,rects,imgsfilename);
        dlib::rectangle dlibrect;
        string winname("Cambridge TIF");
        cv::namedWindow(winname,0);
        for (int i = 0; i < names.size(); ++i)
        {
            cout << "processing image # " << i << " "<< names[i] << endl;
            dlibrect.set_top(rects[i].y);
            dlibrect.set_left(rects[i].x);
            dlibrect.set_right(rects[i].x + rects[i].width);
            dlibrect.set_bottom(rects[i].y + rects[i].height);
            cv::Mat frame;
            frame = cv::imread(names[i]);
            cv::Mat gray;
            array2d<unsigned char> img;
            cv::cvtColor(frame,gray,CV_BGR2GRAY);
            dlib::cv_image<unsigned char> dlibimg(gray);
            assign_image(img, dlibimg);
		    clock_t begin = clock();
            full_object_detection shape = sp(img, dlibrect);
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "FPS: " << 1./elapsed_secs << std::endl;
            for (unsigned int k = 0; k < shape.num_parts(); k++){
                out << shape.part(k).x() << " " << shape.part(k).y()  << " ";
                cv::circle(frame, cv::Point_<int>(shape.part(k).x(),shape.part(k).y()),3, cv::Scalar(0, 0, 255, 0),3);
            }
            cv::rectangle(frame,rects[i],cv::Scalar(128, 128, 0, 0),2);
            out << std::endl;
            cv::imshow(winname,frame);
            cv::waitKey(1000);//To decrease this number for speed up. 
        }
        out.close();
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

