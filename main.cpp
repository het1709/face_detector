#include <string>
#include <cstdlib>
#include "opencv2/opencv.hpp"

const int REQ_WIDTH = 320;

int main (int argc, char** argv) {
    // Load a pretrained LBP detector for frontal face detection
    cv::CascadeClassifier faceDetector;
    try {
        faceDetector.load("lbpcascade_frontalface.xml");
    } catch (cv::Exception e) {
        std::cerr << "Could not load face detector" << std::endl;
        exit(1);
    }
    cv::VideoCapture camera(0); // Access webcam
    // // Set the camera resolution
    // camera.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    // camera.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    while (true) {
        // get next camera frame
        cv::Mat cameraFrame;
        camera >> cameraFrame;

        // Convert frame to grayscale
        cv::Mat gray;
        if (cameraFrame.channels() == 3) {
            // cameraFrame is a BGR image
            cvtColor(cameraFrame, gray, CV_BGR2GRAY);
        } else if (cameraFrame.channels() == 4) {
            // cameraFrame is a BGRA image
            cvtColor(cameraFrame, gray, CV_BGRA2GRAY);
        } else {
            // cameraFrame is already in grayscale
            gray = cameraFrame;
        }

        // Shrink the grayscale image
        cv::Mat smallImg;
        float scale = gray.cols / (float) REQ_WIDTH;
        if (scale > 1) {
            // Shrink image
            int scaledHeight = cvRound(gray.rows / scale);
            resize(gray, smallImg, cv::Size(REQ_WIDTH, scaledHeight));
        } else {
            // Input is already small enough
            smallImg = gray;
        }

        // Perform histogram equalization
        cv::Mat equalizedImg;
        equalizeHist(smallImg, equalizedImg);

        int flags = cv::CASCADE_FIND_BIGGEST_OBJECT | cv::CASCADE_DO_ROUGH_SEARCH; // Search for single face. Change to CASCADE_SCALE_IMAGE to search for many faces
        cv::Size minFeatureSize(80, 80); // Change to 20 x 20 if you need to detect far away faces
        float searchScaleFactor = 1.1f;
        int minNeighbours = 4;

        std::vector<cv::Rect> faces;
        faceDetector.detectMultiScale(equalizedImg, faces, searchScaleFactor, minNeighbours, flags, minFeatureSize);
        std::cout << "Number of faces detected: " << faces.size() << std::endl;
    }
}