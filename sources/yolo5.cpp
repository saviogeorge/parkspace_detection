// Include Libraries.
#include <opencv2/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

#include "../headers/yolo_OBD.hpp"

// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;


extern Image img_obj;


namespace camera_params
{
    // // Open video file
    // VideoCapture video("path to video file.mp4");
    // double fps = video.get(CAP_PROP_FPS);
    // cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps << endl;
    // The above code was used to determine the FPS.
    const unsigned int FPS = 30;
}

bool spot_available_confirmed(int frame_no)
{
    return frame_no>camera_params::FPS*3;
} 


bool parking_space_check(cv::VideoCapture& cap)
{
    Mat frame;
    bool is_success = cap.read(frame);
    YoloOBD obj_det;

    if (is_success)
    {
        // Load model.
        Net net;
        net = readNet("/media/savio/5CA2CA60A2CA3E70/Career/2022/Tech_preparation/projects/parkspace_detection/models/yolov5m.onnx"); 

        // Preprocess the image and get obtain the objects detected.
        vector<Mat> detections;
        detections = obj_det.pre_process(frame, net);


        obj_det.post_process(frame, detections);
        
        // The idea is the bounding boxes of the cars parked next to each other will always 
        // overlap if it does not overlap then we assume there is a potential space for parking
        // in between.
        // Note: Not an optimal approach but works with the current set up.
        // TODO: Alternatively the IoU can be employed with a threshold value

        bool parkspace_detected = !(std::all_of(img_obj.box_intersect_list.begin(),img_obj.box_intersect_list.end(),[](bool v){return v;}));

        return parkspace_detected;
    }
    // If frames are not there, close it
    if (is_success == false)
    {
        cout << "Video camera is disconnected" << endl;
    }        
    // wait 20 ms between successive frames and break the loop if key q is pressed
    int key = waitKey(20);
    if (key == 'q')
    {
        cout << "q key is pressed by the user. Stopping the video" << endl;
    }


}


int main()
{

    cv::VideoCapture cap;
    cap.open("/media/savio/5CA2CA60A2CA3E70/Career/2022/Tech_preparation/projects/parkspace_detection/test_video.mp4");
    int index = 0;
    while(cap.isOpened())
    {
        bool parkspace_detected = parking_space_check(cap);

        while(parkspace_detected)
        {
            imshow("Output", img_obj.img);
            waitKey(1);
            index++;
            cout<<index<<endl;
            while(parkspace_detected&&spot_available_confirmed(index))
            {
                cout<<"THE PARKING SPACE IS AVAILABLE"<<endl;
                parkspace_detected = parking_space_check(cap);
            }
            parkspace_detected = parking_space_check(cap);
        }
        index = 0;

        imshow("Output", img_obj.img);
        waitKey(1);
            
    }
    cap.release();
    destroyAllWindows();

    return 0;
}




