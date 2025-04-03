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
cv::Point objectPos(50 , 25);
bool isDragging = false;
cv::Point clickOffset;

void mouseCallBack(int event, int x, int y, int flags, void* userdata) {
    
    int boxWidth = 92 * 3;
    int boxHeight = 112 * 3;
    int frameWidth = 640;  
    int frameHeight = 480;

    if (userdata) {
        cv::Size* frameSize = static_cast<cv::Size*>(userdata);
        frameWidth = frameSize->width;
        frameHeight = frameSize->height;
    }

    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        if (cv::Rect(objectPos, cv::Size(92*3, 112*3)).contains(cv::Point(x, y))) {
            isDragging = true;
            clickOffset = objectPos - cv::Point(x, y);
            //std::cout << "Click down is registered." << std::endl;
        }
        break;

    case cv::EVENT_MOUSEMOVE:
        if (isDragging) {
            objectPos = cv::Point(x, y) + clickOffset;
            //std::cout << "Click Move is registered.";

            objectPos.x = std::max(0, std::min(objectPos.x, frameWidth - boxWidth));
            objectPos.y = std::max(0, std::min(objectPos.y, frameHeight - boxHeight));
        }
        break;

    case cv::EVENT_LBUTTONUP:
        isDragging = false;
        //std::cout << "Click release is registered.";
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

  cv::namedWindow(win_name);
  cv::setMouseCallback(win_name, mouseCallBack);

  std::cout << "Wait up to 60 secs. for camera access to be obtained..." << std::endl;
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
    std::cout << "training complete! put your face in the box!" << std::endl;


  if (vid_in.isOpened())
  {
      std::cout << "Camera capture obtained!" << std::endl;
  }
  else
  {
      std::cerr << "error: Camera 0 could not be opened for capture.\n";
      return -1;
  }


  int i{ 0 }; // counter to save multiple images
  while (1) {
      vid_in >> frame;

      int frameWidth = frame.cols;
      int frameHeight = frame.rows;
      int boxWidth = 92 * 3;  
      int boxHeight = 112 * 3; 
      int x = (frameWidth - boxWidth) / 2;  
      int y = (frameHeight - boxHeight) / 3; 
      

      cv::rectangle(frame, objectPos,objectPos + cv::Point(boxWidth,boxHeight), cv::Scalar(255, 200, 100), 2);
      // after frame used to be -> cv::Point(x, y)

      int code = cv::waitKey(1000 / fps); // how long to wait for a key (msecs)
      if (code == 27) // escape
      { 
          break;
      
      }

      cv::Mat grabbedImage = frame(cv::Rect(objectPos, cv::Size( boxWidth, boxHeight))).clone();
      cv::cvtColor(grabbedImage, grabbedImage, cv::COLOR_BGR2GRAY); //converts to greayscale
      cv::resize(grabbedImage, grabbedImage, cv::Size(92, 112)); //resizes the image to the correct size (the size of the box has to be the correct aspect ratio or it doesnt run)
      

      // flips box upside down and inverts colours //
      /*
      */
      cv::Mat Area = frame(cv::Rect(objectPos, cv::Size(boxWidth, boxHeight))).clone();
      cv::Mat flippedArea, invertedArea;
      cv::flip(Area, flippedArea, 0);
      cv::bitwise_not(flippedArea, invertedArea);
      
      invertedArea.copyTo(frame(cv::Rect(objectPos, cv::Size(boxWidth, boxHeight))));
      cv::imshow(win_name, frame);

      int predictedLabel = model->predict(grabbedImage);

      if (predictedLabel == 41 ||predictedLabel == 42 || predictedLabel == 43) // space.  ""
      {
          if (currentPrediction != predictedLabel) 
          {
                cv::imwrite(std::string("../out") + std::to_string(i++) + ".png", grabbedImage); //saves file to test filters and moving box
                currentPrediction = predictedLabel;

                std::cout << "'\n'**********************************************";
                std::cout << "'\n'*                                            *";
                std::cout << "'\n'*          Predicted class = " << predictedLabel <<"              *";
                std::cout << "'\n'*                                            *";
                std::cout << "'\n'**********************************************";

                //break;
          }


      }
      
  }
  vid_in.release();
  return 0;
}
