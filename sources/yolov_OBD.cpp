#include "../headers/yolo_OBD.hpp"
#include <fstream>

Image img_obj;

namespace blob_params
{

    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float SCORE_THRESHOLD = 0.5;
    const float NMS_THRESHOLD = 0.45;
    const float CONFIDENCE_THRESHOLD = 0.45;
}

namespace text_params
{
    const float FONT_SCALE = 0.7;
    const int FONT_FACE = cv::HersheyFonts::FONT_HERSHEY_SIMPLEX;
    const int THICKNESS = 1;
}

namespace opencv_params
{
    cv::Scalar BLACK = cv::Scalar(0,0,0);
    cv::Scalar BLUE = cv::Scalar(255, 178, 50);
    cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
    cv::Scalar RED = cv::Scalar(0,0,255);
}


/**
* Preprocess and forward the input image to the neural network.
* 
* @param input_image[in] - reference to the input image frame to be processed.
* @param net[in] - reference to the opencv cv::dnn::dnn4
* @returns  -  dnn detection results
The returned object is a 2-D array. 
The output depends on the size of the input. 
For example, with the default input size of 640, 
we get a 2D array of size 25200×85 (rows and columns). 
The rows represent the number of detections. 
So each time the network runs, it predicts 25200 bounding boxes. 
Every bounding box has a 1-D array of 85 entries that tells the quality of the detection. 
This information is enough to filter out the desired detections.
[X|Y|W|H|Confidence|Class scores of 80 classes].
The first two places are normalized center coordinates of the detected bounding box. 
Then comes the normalized width and height. 
Index 4 has the confidence score that tells the probability of 
the detection being an object. The following 80 entries tell the 
class scores of 80 objects of the COCO dataset 2017, on which the model has been trained.   
*/
std::vector<cv::Mat> YoloOBD::pre_process(cv::Mat &input_image, cv::dnn::Net &net)
{
    // Convert to blob.
    cv::Mat blob;
    //https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
    //https://docs.opencv.org/3.4/d6/d0f/group__dnn.html
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(blob_params::INPUT_WIDTH, blob_params::INPUT_HEIGHT), cv::Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}


/**
* Post process the input image and store the results in the outputs. Returns true if successful false otherwise.
* 
* @param input_image - reference to the input image frame to be processed.
* @param outputs - dnn detection results from pre process stage 
*/
void YoloOBD::post_process(cv::Mat &input_image, std::vector<cv::Mat> &outputs)
{
     // Initialize vectors to hold respective outputs while unwrapping detections.
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes; 
    std::vector<std::string> class_name;

    get_classification_list(class_name);

    // Resizing factor.
    float x_factor = input_image.cols / blob_params::INPUT_WIDTH;
    float y_factor = input_image.rows / blob_params::INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    // Iterate through 25200 detections.
    // This method is used to calculate the class scores for the class scores.
    for (int i = 0; i < rows; ++i) 
    {
        float confidence = data[4];
        // filter out the good and discard the bad detections
        // Check if the class score is above the threshold.
        if (confidence >= blob_params::CONFIDENCE_THRESHOLD) 
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            cv::Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            // Check if the class score is less than the threshold.
            if (max_class_score > blob_params::SCORE_THRESHOLD) 
            {
                // Store class ID and confidence in the pre-defined respective vectors.

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += 85;
    }

    // Perform Non Maximum Suppression and draw predictions.
    std::vector<int> indices;
    // Non- Maximum suppression - Algorithm to filter out the best bounding boxes  
    cv::dnn::NMSBoxes(boxes, confidences, blob_params::SCORE_THRESHOLD, blob_params::NMS_THRESHOLD, indices);
    draw_bounding_boxes(input_image, boxes, confidences, indices, class_ids);

    img_obj.img = input_image;
    img_obj.box_intersect_list = intersect_list(boxes, indices);

}

/**
* Draws bounding boxes on the input image. This function is called by Yolov5Model :: draw_predictions ()
* 
* @param input_image
* @param box_list
* @param confidence_list
* @param box_indices
* @param class_ids
*/
void YoloOBD::draw_bounding_boxes(cv::Mat &input_image, std::vector<cv::Rect> &box_list, std::vector<float> &confidence_list, std::vector<int> &box_indices, std::vector<int> &class_ids)
{

    std::vector<bool> intersect_list;
    std::vector<std::string> class_name;
    get_classification_list(class_name);
    // Draw the bounding box.
    for (int i = 0; i < box_indices.size(); i++) 
    {
        int idx = box_indices[i];
        cv::Rect box = box_list[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(input_image, cv::Point(left, top), cv::Point(left + width, top + height), opencv_params::BLUE, 3*text_params::THICKNESS);

        // Get the label for the class name and its confidence.
        std::string label = cv::format("%.2f", confidence_list[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);
    }

}

/**
* Returns a list of boolean. True if the adjacent detected boxes intersect or False.
*
* @param box_list
* @param box_indices
*/
std::vector<bool> YoloOBD::intersect_list(std::vector<cv::Rect> &box_list, std::vector<int> &box_indices)
{

    std::vector<bool> intersect_list;
    // Add boxes to the intersecting box_list.
    for (int i = 0; i < box_indices.size(); i++) 
    {
        int idx1 = box_indices[i];
        cv::Rect box1 = box_list[idx1];
        // Add a box to the intersect list.
        if(i+1<box_indices.size())
        {
            int idx2 = box_indices[i+1];
            cv::Rect box2 = box_list[idx2];
            bool intersects = ((box1&box2).area()>0);
            intersect_list.push_back(intersects);
        }
    }

    return intersect_list;

}

/**
* Get list of classes from coco. names file and store in vector class_list. This is used to determine which classes are present in the model
* 
* @param class_list - vector to store class
*/
void YoloOBD::get_classification_list(std::vector<std::string> &class_list)
{
    std::ifstream ifs("coco.names");
    std::string line;

    // Get the first line in the class list.
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
}

// Draw the predicted bounding box.
/**
* Draws the label on the Yolov5 image. It is assumed that the image is in OpenCV format
* 
* @param input_image - The image to draw on
* @param label - The label to draw on the image. This must be a string
* @param left - The left position of the label
* @param top - The top position of the label ( will be clipped
*/
void YoloOBD::draw_label(cv::Mat& input_image, std::string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, text_params::FONT_FACE, text_params::FONT_SCALE, text_params::THICKNESS, &baseLine);
    top = std::max(top, label_size.height);
    // Top left corner.
    cv::Point tlc = cv::Point(left, top);
    // Bottom right corner.
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw black rectangle.
    rectangle(input_image, tlc, brc, opencv_params::BLACK, cv::FILLED);
    // Put the label on the black rectangle.
    cv::putText(input_image, label, cv::Point(left, top + label_size.height), text_params::FONT_FACE, text_params::FONT_SCALE, opencv_params::YELLOW, text_params::THICKNESS);
}
