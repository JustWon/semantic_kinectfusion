#include <iostream>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <kfusion/kinfu.hpp>
#include <kfusion/cuda/marching_cubes.hpp>
//#include <io/capture.hpp>
#include <io/bin_grabber.hpp>
#include <string>

using namespace kfusion;
using namespace std;


class SequenceSource 
{
    string dataset_dir;
    string asso_file;
    vector<string> color_files;
    vector<string> depth_files;
    float magic_factor;
    int seq_n;
    int idx;

public:
    vector<string> split(const char *str, char c = ' ')
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

    SequenceSource(string _dataset_dir, float _magic_factor)
    {
        dataset_dir = _dataset_dir;
        asso_file = _dataset_dir + "/associations.txt";
        magic_factor = _magic_factor;
        // cout << asso_file << endl;

        ifstream openFile(asso_file.c_str());
        if( openFile.is_open() ){
            string line;
            while(getline(openFile, line)){
                // cout << line << endl;
                vector<string> tokens = split(line.c_str(), ' ');
                color_files.push_back(dataset_dir+tokens[1]);
                depth_files.push_back(dataset_dir+tokens[3]);
            }
            openFile.close();
        }

        idx=0;
        seq_n = color_files.size();
    }

    bool grab(cv::Mat& depth, cv::Mat& color)
    {
        std::string depth_file_name;
        std::string color_file_name;
        color_file_name = color_files[idx];
        depth_file_name = depth_files[idx];

        depth = cv::imread(depth_file_name, CV_LOAD_IMAGE_ANYDEPTH);
        for (int y = 0 ; y < depth.rows ; y++)
        for (int x = 0 ; x < depth.cols ; x++)
        {
            depth.at<ushort>(y,x) *= magic_factor;
        }

        color = cv::imread(color_file_name);
        cv::cvtColor(color, color, CV_BGR2BGRA, 4);

        // cv::imshow("color", color);
        // cv::imshow("depth", depth);
        // cv::waitKey();

        if (idx++ > seq_n)
        {
            idx = 0;
            return false;
        }

        return true;
    }
    void reset()
    {
        idx = 0;
    }
};

struct KinFuApp
{
  /**
   * @name KeyboardCallback
   * @fn static void KeyboardCallback(const cv::viz::KeyboardEvent& event, void* pthis)
   * @brief Define keyboard callback for the Kinfu app
   * @param[in] event
   * @param[in] pthis, void pointer on itself
   */
  static void KeyboardCallback(const cv::viz::KeyboardEvent& event, void* pthis)
  {
      KinFuApp& kinfu = *static_cast<KinFuApp*>(pthis);

      if(event.action != cv::viz::KeyboardEvent::KEY_DOWN)
          return;

      if(event.code == 't' || event.code == 'T')
          kinfu.take_cloud(*kinfu.kinfu_);

      if(event.code == 'i' || event.code == 'I')
          kinfu.interactive_mode_ = !kinfu.interactive_mode_;

      if(event.code == 'm' || event.code == 'M')
          kinfu.take_mesh(*kinfu.kinfu_);
  }

  KinFuApp(SequenceSource& source, const KinFuParams& params) : exit_ (false), capture_(source), interactive_mode_(false), pause_(false) {
    kinfu_ = KinFu::Ptr( new KinFu(params) );

    // capture_.setRegistration(true);

    cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
    viz.showWidget("cube", cube, params.volume_pose);
    viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
    viz.registerKeyboardCallback(KeyboardCallback, this);
  }

  /**
   * @name show_depth
   * @fn void show_depth(const cv::Mat& depth)
   * @brief Display the depth stream, after normalization
   * @param[in] depth, the depth image to normalize and display
   */
  void show_depth(const cv::Mat& depth) {
      cv::Mat display;
      cv::normalize(depth, display, 0, 255, cv::NORM_MINMAX, CV_8U);
      depth.convertTo(display, CV_8U, 255.0/4000);
      cv::imshow("Depth", display);
  }

  /**
   * @name show_raycasted
   * @fn void show_raycasted(KinFu& kinfu)
   * @brief Show the reconstructed scene (using raycasting)
   * @param[in] kinfu instance
   */
  void show_raycasted(KinFu& kinfu) {
      const int mode = 3;
      if (interactive_mode_)
          kinfu.renderImage(view_device_, viz.getViewerPose(), mode);
      else
          kinfu.renderImage(view_device_, mode);

      view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
      view_device_.download(view_host_.ptr<void>(), view_host_.step);
      cv::imshow("Scene", view_host_);
  }

  /**
   * @name take_cloud
   * @fn void take_cloud(KinFu& kinfu)
   * @brief Fetch cloud and display it
   * @param[in] kinfu instance
   */
  void take_cloud(KinFu& kinfu)
  {
      cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer_);

      cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);

      if (kinfu.params().integrate_color)
      {
          kinfu.color_volume()->fetchColors(cloud, color_buffer_);
          cv::Mat color_host(1, (int)cloud.size(), CV_8UC4);
          cloud.download(cloud_host.ptr<Point>());
          color_buffer_.download(color_host.ptr<RGB>());
          viz.showWidget("cloud", cv::viz::WCloud(cloud_host, color_host));
      } else
      {
          cloud.download(cloud_host.ptr<Point>());
          // viz.showWidget("cloud", cv::viz::WCloud(cloud_host));
          viz.showWidget("cloud", cv::viz::WPaintedCloud(cloud_host));
      }
  }

  /**
   * @name take_mesh
   * @fn void take_mesh(KinFu& kinfu)
   * @brief Run marching cubes on the volume and construct the mesh
   * @param[in] kinfu instance
   */
  void take_mesh(KinFu& kinfu)
  {
      if (!marching_cubes_)
          marching_cubes_ = cv::Ptr<cuda::MarchingCubes>(new cuda::MarchingCubes());

      cuda::DeviceArray<Point> triangles = marching_cubes_->run(kinfu.tsdf(), triangles_buffer_);
      int n_vert = triangles.size();

      cv::viz::Mesh mesh;
      mesh.cloud.create(1, n_vert, CV_32FC4);
      mesh.polygons.create(1, 4*n_vert/3, CV_32SC1);

      for (int i = 0; i < n_vert/3; ++i) {
          mesh.polygons.at<int>(4*i) = 3;
          mesh.polygons.at<int>(4*i+1) = 3*i;
          mesh.polygons.at<int>(4*i+2) = 3*i+1;
          mesh.polygons.at<int>(4*i+3) = 3*i+2;
      }

      cv::Mat mesh_colors(1, n_vert, CV_8UC4);

      if (kinfu.params().integrate_color)
      {
          kinfu.color_volume()->fetchColors(triangles, color_buffer_);
          color_buffer_.download(mesh_colors.ptr<RGB>());
          mesh.colors = mesh_colors;
      }

      triangles.download(mesh.cloud.ptr<Point>());

      viz.showWidget("cloud", cv::viz::WMesh(mesh));
  }

  /**
   * @name execute
   * @fn bool execute()
   * @brief Run the main loop of the app
   * @return true if no error, false otherwise
   */
  bool execute()
  {
      KinFu& kinfu = *kinfu_;
      cv::Mat depth, image;
      double time_ms = 0;
      bool has_image = false;

      for (int i = 0; !exit_ && !viz.wasStopped(); ++i)
      {
          bool has_frame = capture_.grab(depth, image);
          if (!has_frame)
              return std::cout << "Can't grab" << std::endl, false;

          depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
          color_device_.upload(image.data, image.step, image.rows, image.cols);

          {
              SampledScopeTime fps(time_ms); (void)fps;
              if (kinfu.params().integrate_color)
                  has_image = kinfu(depth_device_, color_device_);
              else
                  has_image = kinfu(depth_device_);
          }

          if (has_image)
              show_raycasted(kinfu);

          show_depth(depth);
          if (kinfu.params().integrate_color)
              cv::imshow("Image", image);

          if (!interactive_mode_)
              viz.setViewerPose(kinfu.getCameraPose());

          int key = cv::waitKey(pause_ ? 0 : 3);

          switch(key)
          {
          case 't': case 'T' : take_cloud(kinfu); break;
          case 'i': case 'I' : interactive_mode_ = !interactive_mode_; break;
          case 'm': case 'M' : take_mesh(kinfu); break;
          case 27: exit_ = true; break;
          case 32: pause_ = !pause_; break;
          }

          //exit_ = exit_ || i > 100;
          viz.spinOnce(3, true);
      }
      return true;
  }

  SequenceSource& capture_;
  /**< */
  bool pause_; // = false
  /**< Stop the execution when set to true */
  bool exit_;
  /**< Allow for free point of view (otherwise, follows the camera) */
  bool interactive_mode_;
  /**< Pointer to the instance of kinfu */
  KinFu::Ptr kinfu_;
  /**< */
  cv::viz::Viz3d viz;

  /**< View of the scene (raycasting) */
  cv::Mat view_host_;
  /**< View of the scene on the GPU */
  cuda::Image view_device_;
  /**< Depth frame on the GPU */
  cuda::Depth depth_device_;
  /**< Color frame on the GPU */
  cuda::Image color_device_;
  /**< point buffer used when fetching the point cloud from the tsdf volume */
  cuda::DeviceArray<Point> cloud_buffer_;
  /**< color buffer used when fetching the colors from the color volume */
  cuda::DeviceArray<RGB> color_buffer_;

  /**< Marching cubes instance (to generate a MESH) */
  cv::Ptr<cuda::MarchingCubes> marching_cubes_;
  /**< triangles buffer used in marching cubes */
  cuda::DeviceArray<Point> triangles_buffer_;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{
  int device = 1;
  cuda::setDevice (device);
  cuda::printShortCudaDeviceInfo (device);

  // string dataset_dir = "/media/dongwonshin/Ubuntu Data/Datasets/TUM/3D Object Reconstruction/rgbd_dataset_freiburg3_cabinet/rgbd_dataset_freiburg3_cabinet/";
  // float magic_factor = 1;
  // float volume_size = 10.0f;
  // string dataset_dir = "/media/dongwonshin/Ubuntu Data/Datasets/TUM/3D Object Reconstruction/rgbd_dataset_freiburg3_teddy/rgbd_dataset_freiburg3_teddy/";
  // float magic_factor = 1;
  // float volume_size = 10.0f;
  // string dataset_dir = "/media/dongwonshin/Ubuntu Data/Datasets/TUM/3D Object Reconstruction/rgbd_dataset_freiburg1_plant/rgbd_dataset_freiburg1_plant/";
  // float magic_factor = 1;
  // float volume_size = 10.0f;
  string dataset_dir = "/media/dongwonshin/Ubuntu Data/Datasets/ICL-NUIM/living_room_traj0_frei_png/";
  float magic_factor = 0.1;
  float volume_size = 5.0f;

  SequenceSource capture(dataset_dir, magic_factor);

  KinFuParams custom_params = KinFuParams::default_params();
  custom_params.integrate_color = true;
  custom_params.tsdf_volume_dims = Vec3i::all(512);
  custom_params.color_volume_dims = Vec3i::all(512);
  custom_params.volume_size = Vec3f::all(volume_size);
  custom_params.volume_pose = Affine3f().translate(Vec3f(-custom_params.volume_size[0]/2, -custom_params.volume_size[1]/2, 0.5f));
  custom_params.tsdf_trunc_dist = 0.002;
  custom_params.bilateral_kernel_size = 3;     //pixels

  KinFuApp app (capture, custom_params);

  // executing
  try { app.execute (); }
  catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
  catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

  return 0;
}
