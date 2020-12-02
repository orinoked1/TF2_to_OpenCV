#include <windows.h>
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace std;
using namespace dnn;

int main(int argc, char** argv)
{
	// define net path & test file path 
	string netPath = "C:\\git_repos\\TF2_to_OpenCV\\py\\model\\frozenGraph_folder\\frozenGraph.pb";
	// load network 
	Net resNet50 = readNet(netPath);
	// load test image 
	string inputFilePath = "C:\\git_repos\\TF2_to_OpenCV\\py\\img_folder\\test_img.xml";
	FileStorage inputFile;
	Mat inputImage;
	inputFile.open(inputFilePath, FileStorage::READ);
	inputFile["test_img"] >> inputImage;
	inputFile.release();
	// reshape image to blob [rows X cols X channels] ->  [observation(1) X channels X rows X cols]
	Mat blob;
	blobFromImage(inputImage, blob, 1.0, Size(inputImage.cols, inputImage.rows), Scalar(0.0), false, false);
	// forward pass
	Mat  featureVector;
	resNet50.setInput(blob);
	featureVector = resNet50.forward();
	// save output 
	string outputFilePath = "C:\\git_repos\\TF2_to_OpenCV\\py\\img_folder\\cpp_feature_vector.xml";
	FileStorage outputFile;
	outputFile.open(outputFilePath, FileStorage::WRITE);
	outputFile << "feature_vector" << featureVector;
	outputFile.release();
}