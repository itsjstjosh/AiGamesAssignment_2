#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp> //test
#include "opencv2/core/utils/logger.hpp" // utils::logging::LOG_LEVEL_WARNING
#include <iostream>
#include <filesystem>
#include <random>
#include <vector>

// g++ -std=c++17 face-rec.cpp -lopencv_face -lopencv_core -lopencv_imgcodecs

//For mouse drag
cv::Rect rect(100, 100, 100, 100);
bool isDragging = false;
cv::Point clickOffset;

void mouseCallBack(int event, int x, int y, int flags, void* userdata) {
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        if (rect.contains(cv::Point(x, y))) {
            isDragging = true;
            clickOffset = cv::Point(x, y) - rect.tl();
            std::cout << "Click down is registered." << std::endl;
        }
        break;

    case cv::EVENT_MOUSEMOVE:
        if (isDragging) {
            rect.x = x - clickOffset.x;
            rect.y = y - clickOffset.y;
            std::cout << "Click Move is registered." << std::endl;
        }
        break;

    case cv::EVENT_LBUTTONUP:
        isDragging = false;
        std::cout << "Click release is registered." << std::endl;
        break;
    }
    
}

int main(int argc, char *argv[])
{
  namespace fs = std::filesystem;

  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

  std::vector<cv::Mat> images;
  std::vector<int>     labels;
  cv::Mat frame;
  double fps = 30;
  const char win_name[] = "Live Video...";
  int currentPrediction = -1;


  std::cout << "Wait 60 secs. for camera access to be obtained..." << std::endl;
  cv::VideoCapture vid_in(0);   // argument is the camera id

    std::cout << " training..." << std::endl;
    fs::path p(argc > 1 ? argv[1] : "../../att_faces");
    for (const auto &entry : fs::recursive_directory_iterator{ p }) 
    {
        if (fs::is_regular_file(entry.status())) 
        { // Was once always (wrongly) false in VS
            if (entry.path().extension() == ".pgm") {
            std::string str = entry.path().parent_path().stem().string(); // s26 s27 etc.
            int label = atoi(str.c_str() + 1); // s1 -> 1 (pointer arithmetic)
            images.push_back(cv::imread(entry.path().string().c_str(), cv::IMREAD_GRAYSCALE));
            labels.push_back(label);
            }
        }
    }
    cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
    model->train(images, labels);
    std::cout << "training complete" << std::endl;


  if (vid_in.isOpened())
  {
      std::cout << "Camera capture obtained." << std::endl;
  }
  else
  {
      std::cerr << "error: Camera 0 could not be opened for capture.\n";
      return -1;
  }

  int i{ 0 }; // a simple counter to save multiple images
  while (1) {
      vid_in >> frame;

      int frameWidth = frame.cols;
      int frameHeight = frame.rows;
      int boxWidth = 92 * 3;  
      int boxHeight = 112 * 3; 
      int x = (frameWidth - boxWidth) / 2;  
      int y = (frameHeight - boxHeight) / 3; 

      cv::rectangle(frame, cv::Point(x, y), cv::Point(x + boxWidth, y + boxHeight), cv::Scalar(255, 200, 100), 2);
      //setMouseCallback("Draggable Object", mouseCallback); Attempt at calling the mouse clicks

      int code = cv::waitKey(1000 / fps); // how long to wait for a key (msecs)
      if (code == 27) // escape. See http://www.asciitable.com/
      { 
          break;
      
      }

      cv::Mat grabbedImage = frame(cv::Rect(x,y,boxWidth, boxHeight)).clone();
      cv::cvtColor(grabbedImage, grabbedImage, cv::COLOR_BGR2GRAY);
      cv::resize(grabbedImage, grabbedImage, cv::Size(92, 112));
      

      // flips box upside down remove comment lines before submission or for testing //
      /*
      cv::Mat Area = frame(cv::Rect(x, y, boxWidth, boxHeight)).clone();
      cv::Mat flippedArea, invertedArea;
      cv::flip(Area, flippedArea, 0);
      cv::bitwise_not(flippedArea, invertedArea);
      
      invertedArea.copyTo(frame(cv::Rect(x, y, boxWidth, boxHeight)));
      */
      cv::imshow(win_name, frame);

      int predictedLabel = model->predict(grabbedImage);

      if (predictedLabel == 41 ||predictedLabel == 42 || predictedLabel == 43) // space.  ""
      {
          if (currentPrediction != predictedLabel) 
          {
                currentPrediction = predictedLabel;

                std::cout << "'\n'**********************************************";
                std::cout << "'\n'*                                            *";
                std::cout << "'\n'*          Predicted class = " << predictedLabel <<"              *";
                std::cout << "'\n'*                                            *";
                std::cout << "'\n'**********************************************";

                cv::imwrite(std::string("../out") + std::to_string(i++) + ".png", grabbedImage);
                //break;
          }


      }
      
  }


  vid_in.release();
  return 0;



}
