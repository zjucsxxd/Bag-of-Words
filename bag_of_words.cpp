//George Ignatius
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2\ml\ml.hpp"

#include <cstdio>
#include <iostream>
#include<fstream>


using namespace cv;
using namespace std;

vector<Mat> readAllImagesInFolder(string path){
	vector<Mat>Images(20);
	string name;
	for (int i = 1; i <= 9; i++){
		name = path + "0" + to_string(i) + ".jpg";
		Images[i-1]= imread(name);
	}
	for (int i = 10; i <= 20; i++){
		name = path + to_string(i) + ".jpg";
		Images[i - 1] = imread(name);
	}
	return Images;
}


vector<Mat> readAllImagesInFolder_testing(string path){
	vector<Mat>Images(10);
	string name;
	for (int i = 21; i <= 30; i++){
		name = path + to_string(i) + ".jpg";
		Images[i - 21] = imread(name);
	}
	//namedWindow("images", WINDOW_NORMAL);
	//for (int i = 0; i < 20; i++){	
	//imshow("images", Images[i]);
	//cvWaitKey(0);
	//}
	return Images;
}


int main()
{
	string path;
	vector<Mat>AllTraining(100);
	PCA pca;
	
	SIFT sift;
	vector<cv::KeyPoint>imgKeypoints;
	Mat imgDiscriptors;
	vector<cv::KeyPoint>modKeypoints;
	Mat modDiscriptors;
	//int classificationOutput[50];
	ofstream txtfile("results.txt", ios::out | ios::binary | ios::trunc);
	int K[]= { 11, 15, 17, 21 };//neareset neighbours
	int num_components[] = { 15, 20, 30, 50 };//PCA components
	int num_clusters[] = { 50, 100, 200 };
	path = "training/car/image_00";
	vector<Mat>subImages = readAllImagesInFolder(path);
	for (int i = 0; i < 20; i++){
		AllTraining[i] = subImages[i];
	}


	path = "training/cougar/image_00";
	subImages = readAllImagesInFolder(path);
	for (int i = 0; i < 20; i++){
		AllTraining[i + 20] = subImages[i];
	}


	path = "training/face/image_00";
	subImages = readAllImagesInFolder(path);
	for (int i = 0; i < 20; i++){
		AllTraining[i + 40] = subImages[i];
	}


	path = "training/pizza/image_00";
	subImages = readAllImagesInFolder(path);
	for (int i = 0; i < 20; i++){
		AllTraining[i + 60] = subImages[i];
	}

	path = "training/sunflower/image_00";
	subImages = readAllImagesInFolder(path);
	for (int i = 0; i < 20; i++){
		AllTraining[i + 80] = subImages[i];
	}


	//calculating the sift features
	Mat AllTrainingDiscriptors;
	cout << "Calculating SIFT discriptors for all Trainning Images" << "\n";
	cout << "Image 1" << "\n";
	int siftfeatures_per_traingingimage[100];
	sift(AllTraining[0], noArray(), imgKeypoints, imgDiscriptors);
	siftfeatures_per_traingingimage[0] = imgDiscriptors.rows;
	AllTrainingDiscriptors = imgDiscriptors.clone();
	int p = 0;
	for (int i = 1; i < 100; i++){
		cout << "Image " << i + 1 << "\n";
		sift(AllTraining[i], noArray(), imgKeypoints, imgDiscriptors);
		siftfeatures_per_traingingimage[i] = imgDiscriptors.rows;
		AllTrainingDiscriptors.push_back(imgDiscriptors);
	}




	cout << "Reading All Test data" << "\n";
	vector<Mat>AllTestingImages(50);
	path = "testing/car/image_00";
	vector<Mat>subImages_testing = readAllImagesInFolder_testing(path);
	for (int i = 0; i < 10; i++){
		AllTestingImages[i] = subImages_testing[i];
	}
	path = "testing/cougar/image_00";
	subImages_testing = readAllImagesInFolder_testing(path);
	for (int i = 0; i < 10; i++){
		AllTestingImages[i + 10] = subImages_testing[i];
	}

	path = "testing/face/image_00";
	subImages_testing = readAllImagesInFolder_testing(path);
	for (int i = 0; i < 10; i++){
		AllTestingImages[i + 20] = subImages_testing[i];
	}


	path = "testing/pizza/image_00";
	subImages_testing = readAllImagesInFolder_testing(path);
	for (int i = 0; i < 10; i++){
		AllTestingImages[i + 30] = subImages_testing[i];
	}


	path = "testing/sunflower/image_00";
	subImages_testing = readAllImagesInFolder_testing(path);
	for (int i = 0; i < 10; i++){
		AllTestingImages[i + 40] = subImages_testing[i];
	}

	cout << "calculating Sift features for all test images" << "\n" << "\n";
	Mat testing_sift_features;
	int testing_sift_features_lengths[50];
	sift(AllTestingImages[0], noArray(), imgKeypoints, imgDiscriptors);
	testing_sift_features_lengths[0] = imgDiscriptors.rows;
	testing_sift_features = imgDiscriptors.clone();
	for (int i = 1; i < 50; i++){
		sift(AllTestingImages[i], noArray(), imgKeypoints, imgDiscriptors);
		testing_sift_features.push_back(imgDiscriptors);
		testing_sift_features_lengths[i] = imgDiscriptors.rows;
	}


	//Tranning using k nearest neighbours
	Mat trainClasses;
	for (int i = 0; i < 20; i++){
		trainClasses.push_back(0);
	}

	for (int i = 20; i < 40; i++){
		trainClasses.push_back(1);
	}

	for (int i = 40; i < 60; i++){
		trainClasses.push_back(2);
	}

	for (int i = 60; i < 80; i++){
		trainClasses.push_back(3);
	}
	for (int i = 80; i < 100; i++){
		trainClasses.push_back(4);
	}
	



	for (int num_cluster_index = 0; num_cluster_index < 3; num_cluster_index++){
		for (int pca_index = 0; pca_index < 4; pca_index++){
			for (int K_index = 0; K_index < 4; K_index++){
				cout << "PCA on Trainning" << "\n";
				pca(AllTrainingDiscriptors, Mat(), CV_PCA_DATA_AS_ROW, num_components[pca_index]);
				Mat pca_sift;
				pca.project(AllTrainingDiscriptors, pca_sift);

				cout << "K means Clustering" << "\n";
				Mat labels;
				kmeans(pca_sift, num_clusters[num_cluster_index], labels, TermCriteria(CV_TERMCRIT_ITER, 100, 1), 1, KMEANS_PP_CENTERS);
				//cout << labels;

				cout << "Finding the average feature vector for each cluster" << "\n";
				Mat cluster_repesentative(num_clusters[num_cluster_index], num_components[pca_index], CV_32F);
				int num_per_cluster = 0;
				for (int i = 0; i <= num_clusters[num_cluster_index] - 1; i++){
					num_per_cluster = 0;
					Mat temp = Mat::zeros(1, num_components[pca_index], CV_32F);
					for (int j = 0; j < labels.rows; j++){
						if (labels.at<int>(j) == i){
							add(temp, pca_sift.row(j), temp);
							num_per_cluster++;
						}
					}
					cluster_repesentative.row(i) = temp / num_per_cluster;
				}
				//cout << cluster_repesentative;

				cout << "Historgram calculation on Training data" << "\n";
				//calculating the histograms of the training images to form the new feature space
				p = 0;
				Mat Hist_training = Mat::zeros(100, num_clusters[num_cluster_index], CV_32F);
				for (int i = 0; i < 100; i++){
					for (int j = 0; j < siftfeatures_per_traingingimage[i]; j++){
						Hist_training.at<float>(i, labels.at<int>(p + j))++;
					}
					p += siftfeatures_per_traingingimage[i];
				}
				//cout << Hist_training;
				CvKNearest knn;
				knn.train(Hist_training, trainClasses, Mat(), false, K[K_index]);
				//knn2 = knn;
				//cout << "Train class" << "\n";
				//cout << trainClasses;
				cout << "End of Training, Moving on to testing" << "\n";
				//Now we Will read each image and classify it immedialty


				Mat Hist_testing = Mat::zeros(50, num_clusters[num_cluster_index], CV_32F);
				p = 0;
				for (int i = 0; i < 50; i++){
					Mat pca_sift_testing;
					BFMatcher matcher(NORM_L2, false);
					vector<DMatch> matches;
					//Assigning a Lable to the pca sift feature based on the Euclidian distance for the centroids of the training clusters in the SIFT_PCA feature space
					//sift(AllTestingImages[i], noArray(), imgKeypoints, imgDiscriptors);
					pca.project(testing_sift_features.rowRange(Range(p, p + testing_sift_features_lengths[i])), pca_sift_testing);
					p += testing_sift_features_lengths[i];
					matcher.match(pca_sift_testing, cluster_repesentative, matches);

					//Creating the Histogram for the Test image
					for (int j = 0; j < pca_sift_testing.rows; j++){
						Hist_testing.at<float>(i, matches[j].trainIdx)++;
					}
					cout << "New hist_testing" << "\n";
					cout << Hist_testing.row(i) << "\n";
				}
				Mat Classification_result;
				Mat neighborResponses;
				Mat dists;
				cout << "finding Nearest Neighbour for all the histograms" << "\n";
				knn.find_nearest(Hist_testing, K[K_index], Classification_result, neighborResponses, dists);
				cout << Classification_result;
				int class0Correct = 10, class1Correct = 10, class2Correct = 10, class3correct = 10, class4correct = 10;
				int total_accuracy = 50;
				double total_accuracy_percentage, class0CorrectPercentage, class1CorrectPercentage, class2CorrectPercentage, class3CorrectPercentage, class4CorrectPercentage;
				//cout << Classification_result;
				for (int i = 0; i < 50; i++){
					if (((int)Classification_result.at<float>(i)) != (i / 10)){
						if (i / 10 == 0){
							class0Correct--;
						}
						else if (i / 10 == 1){
							class1Correct--;
						}
						else if (i / 10 == 2){
							class2Correct--;
						}
						else if (i / 10 == 3){
							class3correct--;
						}
						else  if (i / 10 == 4){
							class4correct--;
						}
					}
				}
				total_accuracy = class0Correct + class1Correct + class2Correct + class3correct + class4correct;
				total_accuracy_percentage = (total_accuracy / 50.0) * 100.0;
				class0CorrectPercentage = (class0Correct / 10.0)*100.0;
				class1CorrectPercentage = (class1Correct / 10.0)*100.0;
				class2CorrectPercentage = (class2Correct / 10.0)*100.0;
				class3CorrectPercentage = (class3correct / 10.0)*100.0;
				class4CorrectPercentage = (class4correct / 10.0)*100.0;
				txtfile << "Num of Clusters = " << num_clusters[num_cluster_index] << "," << "Num of Nrearest Neighbours = " << K[K_index] << "," << "Num of PCA Components = " << num_components[pca_index] << "\n";
				txtfile << "Total accuracy Percentage = " << total_accuracy_percentage << "\n";
				txtfile << "Car Class accuracy Percentage = " << class0CorrectPercentage << "\n";
				txtfile << "Cougar accuracy Percentage = " << class1CorrectPercentage << "\n";
				txtfile << "Face accuracy Percentage = " << class2CorrectPercentage << "\n";
				txtfile << "Pizza accuracy Percentage = " << class3CorrectPercentage << "\n";
				txtfile << "SunFlower accuracy Percentage = " << class4CorrectPercentage << "\n";
				cout << "Succesfully ran one set of values" << "\n";
			}
		}
	}


	

	txtfile.close();
	

	return 0;
}







