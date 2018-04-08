#include <iostream>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <kfusion/kinfu.hpp>
#include <kfusion/cuda/marching_cubes.hpp>

#include <string>
#include <thread>

#include <DW_Utility.h>

using namespace kfusion;
using namespace std;

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

      if(event.code == 'm')
          kinfu.take_mesh(*kinfu.kinfu_, false);
      if(event.code == 'M')
    	  kinfu.take_mesh(*kinfu.kinfu_, true, "mesh/color_mesh " + kinfu.capture_.current_timestamp() + ".ply");

      if(event.code == 's')
          kinfu.take_semantic_mesh(*kinfu.kinfu_, false);
      if(event.code == 'S')
          kinfu.take_semantic_mesh(*kinfu.kinfu_, true, "mesh/semantic_mesh " + kinfu.capture_.current_timestamp() + ".ply");
  }

  KinFuApp(DW_Utility::SequenceSource& source, const KinFuParams& params) : exit_ (false), capture_(source), interactive_mode_(false), pause_(false) {
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
  void take_mesh(KinFu& kinfu, bool save_mesh, string mesh_string = "color_mesh.ply", bool last_frame=false, bool transformed=false)
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

      if (save_mesh)
      {
    	  std::thread myThread;
    	  if (!transformed)
    		  myThread = std::thread(DW_Utility::outputMeshAsPly, mesh_string, mesh, 1);

		  if (!last_frame)
			  myThread.detach();
		  else // if it is the last frame, we should wait
			  myThread.join();
      }
  }

  void take_semantic_mesh(KinFu& kinfu, bool save_mesh, string mesh_string = "semantic_mesh.ply")
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
          kinfu.semantic_volume()->fetchSemantics(triangles, color_buffer_);
          color_buffer_.download(mesh_colors.ptr<RGB>());
          mesh.colors = mesh_colors;
      }

      triangles.download(mesh.cloud.ptr<Point>());

      viz.showWidget("cloud", cv::viz::WMesh(mesh));

      if (save_mesh)
      {
		  std::thread myThread(DW_Utility::outputMeshAsPly, mesh_string, mesh, 1);
		  myThread.detach();
      }

      // cv::imshow("mesh_colors", mesh_colors);
      // cv::waitKey(0);
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
      cv::Mat depth, image, semantic;
      double time_ms = 0;
      int kinfu_return_val= 0;

      for (int i = 0; !exit_ && !viz.wasStopped(); ++i)
      {
          if (!capture_.grab(depth, image, semantic))
          {
        	   DW_Utility::consolePrint("green", "[End of frames]");
        	   kinfu.saveEstimatedTrajectories();
               take_mesh(kinfu, true, "mesh/color_mesh [finish].ply", true);
               return false;
          }

          depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
          color_device_.upload(image.data, image.step, image.rows, image.cols);
          semantic_device_.upload(semantic.data, semantic.step, semantic.rows, semantic.cols);

		  SampledScopeTime fps(time_ms); (void)fps;
		  kinfu_return_val = kinfu(depth_device_, color_device_, semantic_device_, capture_.current_timestamp());

          // volume save
          if (kinfu_return_val == 1)		// tracking success
          {
        	  show_raycasted(kinfu);
        	  if (kinfu.getFrameCounter() % 100 == 0)
        	  {
        		  take_mesh(kinfu, true, "mesh/color_mesh " + capture_.current_timestamp() + "[split].ply");
//				  kinfu.storeSubvolume();
//				  kinfu.storePoseVector();
				  kinfu.reset();
        	  }
          }
          else if (kinfu_return_val == 2)	// tracking failure,
          {
        	  if (kinfu.getFrameCounter() > 50)	// the number of frame until the failure point should be bigger than 50.
        	  {
        		  take_mesh(kinfu, true, "mesh/color_mesh " + capture_.current_timestamp() + "[faliure].ply");
//				  kinfu.storeSubvolume();
//				  kinfu.storePoseVector();
        	  }
        	  kinfu.reset();
          }
          // else if (inconsistent model)


          if (!interactive_mode_)
              viz.setViewerPose(kinfu.getCameraPose());

          int key = cv::waitKey(pause_ ? 0 : 1);
          switch(key)
          {
			  case 't': case 'T' : take_cloud(kinfu); break;
			  case 'i': case 'I' : interactive_mode_ = !interactive_mode_; break;
			  case 'm': take_mesh(kinfu, false); break;
			  case 'M': take_mesh(kinfu, true, "mesh/color_mesh " + capture_.current_timestamp() + ".ply"); break;
			  case 's': take_semantic_mesh(kinfu, false); break;
			  case 'S': take_semantic_mesh(kinfu, true, "mesh/semantic_mesh " + capture_.current_timestamp() + ".ply"); break;
			  case 27:
				  kinfu.saveEstimatedTrajectories();
				  exit_ = true;
				  break;
			  case 32: pause_ = !pause_; break;
          }

          //exit_ = exit_ || i > 100;
          viz.spinOnce(1, true);
      }
      return true;
  }

  DW_Utility::SequenceSource& capture_;
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
  cuda::Image semantic_device_;

  cuda::DeviceArray<Point> cloud_buffer_;
  /**< color buffer used when fetching the colors from the color volume */
  cuda::DeviceArray<RGB> color_buffer_;

  /**< Marching cubes instance (to generate a MESH) */
  cv::Ptr<cuda::MarchingCubes> marching_cubes_;
  /**< triangles buffer used in marching cubes */
  cuda::DeviceArray<Point> triangles_buffer_;

  bool model_after_fail = false;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{

  if (argc < 2)
  {
	  DW_Utility::consolePrint("red", "[Error] ");
  	  cout << "Config file is necessary." << endl;
  	  exit(0);
  }

  DW_Utility::ConfigParser config(argv[1]);

  cuda::setDevice (config.gpu_id);
  cuda::printShortCudaDeviceInfo (config.gpu_id);

  string dataset_dir = config.dataset_dir;
  float magic_factor = config.magic_factor;
  float volume_size = config.volume_size;
  int img_cols = config.img_cols;
  int img_rows = config.img_rows;
  float focal_length = config.focal_length;
  int start_frame = config.start_frame;

  DW_Utility::SequenceSource capture(dataset_dir, magic_factor, start_frame);

  KinFuParams custom_params = KinFuParams::default_params();
  custom_params.integrate_color = true;
  custom_params.integrate_semantic = true;
  custom_params.tsdf_volume_dims = Vec3i::all(512);
  custom_params.color_volume_dims = Vec3i::all(512);
  custom_params.semantic_volume_dims = Vec3i::all(512);
  custom_params.volume_size = Vec3f::all(volume_size);
  custom_params.volume_pose = Affine3f().translate(Vec3f(-custom_params.volume_size[0]/2, -custom_params.volume_size[1]/2, 0.0f));
  custom_params.tsdf_trunc_dist = 0.002;
  custom_params.bilateral_kernel_size = 3;     //pixels
  custom_params.cols = img_cols;
  custom_params.rows = img_rows;
  custom_params.intr = Intr(focal_length, focal_length, custom_params.cols/2 - 0.5f, custom_params.rows/2 - 0.5f);
  custom_params.icp_dist_thres = 0.25;                //meters
  custom_params.icp_angle_thres = deg2rad(30); //radians

  KinFuApp app (capture, custom_params);

  // executing
  try { app.execute (); }
  catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
  catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

  return 0;
}
