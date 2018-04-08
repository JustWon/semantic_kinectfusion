#include <DW_Utility.h>

namespace DW_Utility {
	void consolePrint(std::string color, std::string text)
	{
		// https://stackoverflow.com/questions/2616906/how-do-i-output-coloured-text-to-a-linux-terminal
		if (color == "red")
			cout << "\033[1;31m" << text  << "\033[0m";
		else if (color == "green")
			cout << "\033[1;32m" << text  << "\033[0m";
		else if (color == "yellow")
			cout << "\033[1;33m" << text  << "\033[0m";
		else if (color == "blue")
			cout << "\033[1;34m" << text  << "\033[0m";
		else if (color == "magenta")
			cout << "\033[1;35m" << text  << "\033[0m";
		else if (color == "cyan")
			cout << "\033[1;36m" << text  << "\033[0m"; 
		else if (color == "white")
			cout << "\033[1;37m" << text  << "\033[0m";
	}

	ConfigParser::ConfigParser(string config_file)
	{
		boost::property_tree::ini_parser::read_ini(config_file, pt);

		gpu_id = atoi(pt.get<std::string>("parameters.gpu_id").c_str());
		start_frame = atof(pt.get<std::string>("parameters.start_frame").c_str());
		dataset_dir = pt.get<std::string>("parameters.dataset_dir");
		magic_factor = atof(pt.get<std::string>("parameters.magic_factor").c_str());
		volume_size = atof(pt.get<std::string>("parameters.volume_size").c_str());
		img_cols = atoi(pt.get<std::string>("parameters.img_cols").c_str());
		img_rows = atoi(pt.get<std::string>("parameters.img_rows").c_str());
		focal_length = atoi(pt.get<std::string>("parameters.focal_length").c_str());

		cout << "[Parameters]" << endl;
		cout << "GPU_ID : \t" << gpu_id << endl;
		cout << "Start frame : \t" <<  start_frame << endl;
		cout << "Dataset dir : \t" <<  dataset_dir << endl;
		cout << "Magic factor: \t" <<  magic_factor << endl;
		cout << "Volume size : \t" <<  volume_size << endl;
		cout << "Img cols : \t" <<  img_cols << endl;
		cout << "Img rows : \t" <<  img_rows << endl;
		cout << "Focal length : \t" <<  focal_length << endl;
	}

	vector<string> SequenceSource::split(const char *str, char c)
	{
		vector<string> result;

		do
		{
			const char *begin = str;

			while(*str != c && *str)
				str++;

			result.push_back(string(begin, str));
		} while (0 != *str++);

		return result;
	}

	SequenceSource::SequenceSource(string _dataset_dir, float _magic_factor, int start_frame)
	{
		dataset_dir = _dataset_dir;
		asso_file = _dataset_dir + "/associations.txt";
		magic_factor = _magic_factor;

		ifstream openFile(asso_file.c_str());
		if( openFile.is_open() ){
			string line;
			while(getline(openFile, line)){
				// cout << line << endl;
				vector<string> tokens = split(line.c_str(), ' ');
				color_files.push_back(dataset_dir+tokens[1]);
				depth_files.push_back(dataset_dir+tokens[3]);
				semantic_files.push_back(dataset_dir+tokens[5]);
				timestamps.push_back(tokens[0]);
			}
			openFile.close();
		}

		idx=start_frame;
		seq_n = color_files.size();
	}

	bool SequenceSource::grab(cv::Mat& depth, cv::Mat& color, cv::Mat& semantic)
	{
		// end of frames
		if (idx >= seq_n)
		{
			idx = 0;
			return false;
		}

		std::string depth_file_name;
		std::string color_file_name;
		std::string semantic_file_name;
		color_file_name = color_files[idx];
		depth_file_name = depth_files[idx];
		semantic_file_name = semantic_files[idx];

		// std::cout << color_file_name << std::endl;
		// std::cout << depth_file_name << std::endl;
		// std::cout << semantic_file_name << std::endl;

		depth = cv::imread(depth_file_name, CV_LOAD_IMAGE_ANYDEPTH);
		for (int y = 0 ; y < depth.rows ; y++)
		for (int x = 0 ; x < depth.cols ; x++)
		{
			depth.at<ushort>(y,x) *= magic_factor;
		}

		color = cv::imread(color_file_name);
		cv::cvtColor(color, color, CV_BGR2BGRA, 4);

		semantic = cv::imread(semantic_file_name);
		cv::cvtColor(semantic, semantic, CV_BGR2BGRA, 4);

		// cv::imshow("color",color);
		// cv::imshow("depth",depth);
		// cv::imshow("semantic",semantic);

		idx++;

		return true;
	}
	void SequenceSource::reset()
	{
		idx = 0;
	}

	string SequenceSource::current_timestamp()
	{
		return timestamps[idx-1];
	}



	void outputMeshAsPly(const std::string& filename, const cv::viz::Mesh& mesh, const cv::Affine3f last_pose, int file_type)
	{
	    if (file_type == 0){
	    	std::ofstream stream(filename.c_str());
	    	size_t num_points = mesh.cloud.cols; // the number of points in the mesh
	    	stream << "ply" << std::endl;
	    	stream << "format ascii 1.0" << std::endl;
	    	stream << "element vertex " << num_points << std::endl;
	    	stream << "property float x" << std::endl;
	    	stream << "property float y" << std::endl;
	    	stream << "property float z" << std::endl;

	    	stream << "property uchar red" << std::endl;
	    	stream << "property uchar green" << std::endl;
	    	stream << "property uchar blue" << std::endl;

	        stream << "element face " << mesh.cloud.cols / 3 << std::endl;
	        stream << "property list uchar int vertex_index" << std::endl;

	        stream << "end_header" << std::endl;

	    	char temp[100];
			for (int i = 0 ; i < mesh.cloud.cols ; i++) {
				cv::Affine3f::Vec3 point(mesh.cloud.at<float>(4*i),mesh.cloud.at<float>(4*i+1),mesh.cloud.at<float>(4*i+2));
				point = last_pose*point;

				stream << point(0) << " " << point(1) << " " << point(2) << " ";

				sprintf(temp, "%d %d %d", mesh.colors.at<unsigned char>(4*i+2),
										  mesh.colors.at<unsigned char>(4*i+1),
										  mesh.colors.at<unsigned char>(4*i));
				stream << temp << endl;
			}
			for (int i = 0 ; i < mesh.cloud.cols/3 ; i++) {
				sprintf(temp, "3 %d %d %d",  3*i+2, 3*i+1, 3*i+0);
				stream << temp << endl;
			}
			stream.close();
	    }
	    else {
	    	std::ofstream stream(filename.c_str() , ios::binary);
			size_t num_points = mesh.cloud.cols; // the number of points in the mesh
			stream << "ply" << std::endl;
			stream << "format binary_little_endian 1.0" << std::endl;
			stream << "element vertex " << num_points << std::endl;
			stream << "property float x" << std::endl;
			stream << "property float y" << std::endl;
			stream << "property float z" << std::endl;

			stream << "property uchar red" << std::endl;
			stream << "property uchar green" << std::endl;
			stream << "property uchar blue" << std::endl;

			stream << "element face " << mesh.cloud.cols / 3 << std::endl;
			stream << "property list uchar int vertex_index" << std::endl;

			stream << "end_header" << std::endl;

			char temp[100];
			for (int i = 0 ; i < mesh.cloud.cols ; i++) {
				cv::Affine3f::Vec3 point(mesh.cloud.at<float>(4*i),
										 mesh.cloud.at<float>(4*i+1),
										 mesh.cloud.at<float>(4*i+2));
				point = last_pose*point;
				stream.write(reinterpret_cast<const char*>(&point(0)), sizeof(float));
				stream.write(reinterpret_cast<const char*>(&point(1)), sizeof(float));
				stream.write(reinterpret_cast<const char*>(&point(2)), sizeof(float));

				stream.write(reinterpret_cast<const char*>(&mesh.colors.at<unsigned char>(4*i+2)), sizeof(char));
				stream.write(reinterpret_cast<const char*>(&mesh.colors.at<unsigned char>(4*i+1)), sizeof(char));
				stream.write(reinterpret_cast<const char*>(&mesh.colors.at<unsigned char>(4*i)), sizeof(char));
			}
			for (int i = 0 ; i < mesh.cloud.cols/3 ; i++) {
				stream.write(reinterpret_cast<const char*>(&mesh.polygons.at<int>(4*i)), sizeof(char));
				stream.write(reinterpret_cast<const char*>(&mesh.polygons.at<int>(4*i+3)), sizeof(int));
				stream.write(reinterpret_cast<const char*>(&mesh.polygons.at<int>(4*i+2)), sizeof(int));
				stream.write(reinterpret_cast<const char*>(&mesh.polygons.at<int>(4*i+1)), sizeof(int));
			}
			stream.close();
	    }
	}

}
