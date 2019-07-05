
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
#define _CRT_SECURE_NO_WARNINGS
#include <Windows.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <chrono>
#include <vector>
#include <thread>


// OPENCV Includes
#include <opencv2/core.hpp>           
#include <opencv2/highgui.hpp>     
#include <opencv2/imgproc.hpp>  

#include "create_blur.h"
#include "file_parser.h"
#include "num2string.h"

#define MAX_CLASSES 256

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	uint32_t idx, jdx, kdx;
    uint32_t row = 0;
    uint32_t col = 0;
    uint32_t index = 0;
	
    double minSigma = 0.32;
	double maxSigma = 2.56 + minSigma;
	
	cv::Size InputImageSize;
	cv::Mat InputImage;
	cv::Mat GroundTruth;
	cv::Mat BlurMap;
	std::vector<cv::Mat> InputImage_RGB(3);
    std::vector<cv::Mat> BlurMap_RGB(3);
    std::vector<cv::Mat> GauBlurR;
    std::vector<cv::Mat> GauBlurG;
    std::vector<cv::Mat> GauBlurB;
	
    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

    std::string data_directory;
	std::string input_image;
	std::string groundtruth_image;
	//std::string image_location;
	std::string blur_image;
	std::string ext;

    std::string blur_type = "_lin_";

	std::vector<int32_t> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);

	std::vector<std::vector<std::string>> params;

	std::cout << "Reading Inputs... " << std::endl;

	//Options = getOptions(argc, argv, "h:f:d");

	std::string parseFilename = argv[1];
    //std::string dataPath = argv[2];

    try
    {
        std::cout << "Parsing inputs from: " << parseFilename << std::endl;

        // parse through the supplied csv file
        parse_csv_file(parseFilename, params);

        // the first entry is the deepest common root directory where the images are stored
        data_directory = params[0][0];
        params.erase(params.begin());

        std::cout << "Images to parse: " << params.size() << std::endl;

        for (kdx = 0; kdx < params.size(); kdx++)
        {
            std::cout << "-------------------------------------------------------------------" << std::endl;
            start_time = chrono::system_clock::now();

            // get the name of the input image without the extension
            groundtruth_image = data_directory + params[kdx][1];
            maxSigma = std::stod(params[kdx][2]) + minSigma;

            //  read in ground truth image
            std::cout << "Reading ground truth image: " << groundtruth_image << std::endl;
            GroundTruth = imread(groundtruth_image, CV_LOAD_IMAGE_GRAYSCALE);

            // read infocus color image 
            input_image = data_directory + params[kdx][0];
            get_file_parts(input_image, blur_image, ext);
            std::string fmt = "%0.2f";
            blur_image = blur_image + blur_type + num2str(minSigma, fmt) + "_" + num2str(maxSigma, fmt) + ".png";

            std::cout << "Reading input image:        " << input_image << std::endl;
            InputImage = imread(input_image, CV_LOAD_IMAGE_COLOR);
            InputImageSize = InputImage.size();

            // split the in focus color image into RGB channel (integers)
            split(InputImage, InputImage_RGB);

            // convert RGB channels to floating point 
            InputImage_RGB[0].convertTo(InputImage_RGB[0], CV_64FC1, 1.0);
            InputImage_RGB[1].convertTo(InputImage_RGB[1], CV_64FC1, 1.0);
            InputImage_RGB[2].convertTo(InputImage_RGB[2], CV_64FC1, 1.0);

            //   Create defocus image for R,G,B channels
            BlurMap = cv::Mat::zeros(InputImageSize, CV_64FC3);
            BlurMap_RGB[0] = cv::Mat::zeros(InputImageSize, CV_64F);
            BlurMap_RGB[1] = cv::Mat::zeros(InputImageSize, CV_64F);
            BlurMap_RGB[2] = cv::Mat::zeros(InputImageSize, CV_64F);

            // Create 256 levels of Gaussian parameters sigma
            // The range is from 0 to MaxSigma
            // Based on 256 different sigma, generate 256 blur image (whole image has the same sigma)//
            std::cout << "Generating Blur Levels... ";
            GauBlurR.clear();
            GauBlurG.clear();
            GauBlurB.clear();
            std::thread t_R(create_blur, InputImage_RGB[0], maxSigma, minSigma, MAX_CLASSES, std::ref(GauBlurR));
            std::thread t_G(create_blur, InputImage_RGB[1], maxSigma, minSigma, MAX_CLASSES, std::ref(GauBlurG));
            std::thread t_B(create_blur, InputImage_RGB[2], maxSigma, minSigma, MAX_CLASSES, std::ref(GauBlurB));
            t_R.join();
            t_G.join();
            t_B.join();

            std::cout << "Generation Complete!" << std::endl;

            // Based on the value of each pixel in ground truth map,
            // Find out which blur index each pixel belongs to
            col = InputImage.cols;
            row = InputImage.rows;

            for (idx = 0; idx < row; idx++)
            {
                for (jdx = 0; jdx < col; jdx++)
                {
                    index = MAX_CLASSES - GroundTruth.at<uint8_t>(idx, jdx) - 1;

                    BlurMap_RGB[0].at<double>(idx, jdx) = GauBlurR[index].at<double>(idx, jdx);
                    BlurMap_RGB[1].at<double>(idx, jdx) = GauBlurG[index].at<double>(idx, jdx);
                    BlurMap_RGB[2].at<double>(idx, jdx) = GauBlurB[index].at<double>(idx, jdx);

                }
            }

            // Combine the three channels and save the defocus image
            merge(BlurMap_RGB, BlurMap);

            std::cout << "Writing blur mapping to file: " << blur_image << std::endl << std::endl;
            imwrite(blur_image, BlurMap, compression_params);

            stop_time = chrono::system_clock::now();
            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

            std::cout << "Completed in " << elapsed_time.count() << " seconds" << std::endl;

        }

        std::cout << "-------------------------------------------------------------------" << std::endl;
        std::cout << "Operations Complete!" << std::endl;

    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }

    std::cout << "Press enter to close..." << std::endl;
    std::cin.ignore();

	return(0);
	
}	// end of main

