#pragma once

#include <iostream>
#include <string>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>

using namespace std;

namespace DW_Utility {
	void consolePrint(std::string color, std::string text);

	class ConfigParser
	{
		boost::property_tree::ptree pt;

	public:
		string dataset_dir;
		double magic_factor;
		int seq_n;
		int idx;
		int gpu_id;
		int start_frame;
		double volume_size;
		int img_cols;
		int img_rows;
		double focal_length;

		ConfigParser(string config_file);
	};


	class SequenceSource
	{
	    string dataset_dir;
	    string asso_file;
	    vector<string> color_files;
	    vector<string> depth_files;
	    vector<string> semantic_files;
	    vector<string> timestamps;
	    float magic_factor;
	    int seq_n;
	    int idx;

	public:
	    vector<string> split(const char *str, char c = ' ');
	    SequenceSource(string _dataset_dir, float _magic_factor, int start_frame);
	    bool grab(cv::Mat& depth, cv::Mat& color, cv::Mat& semantic);
	    void reset();
	    string current_timestamp();
	};

	void outputMeshAsPly(const std::string& filename, const cv::viz::Mesh& mesh, const cv::Affine3f last_pose, int file_type);
}
