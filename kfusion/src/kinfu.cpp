#include "precomp.hpp"
#include "internal.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>

#include <fstream>
using std::ofstream;

using namespace std;
using namespace kfusion;
using namespace kfusion::cuda;
using namespace g2o;

static inline float deg2rad (float alpha) { return alpha * 0.017453293f; }

kfusion::KinFuParams kfusion::KinFuParams::default_params()
{
    const int iters[] = {10, 5, 4, 0};
    const int levels = sizeof(iters)/sizeof(iters[0]);

    KinFuParams p;

    p.cols = 640;  //pixels
    p.rows = 480;  //pixels
    p.integrate_color = false;
    p.integrate_semantic = true;
    p.intr = Intr(525.f, 525.f, p.cols/2 - 0.5f, p.rows/2 - 0.5f);

    p.tsdf_volume_dims = Vec3i::all(512);  //number of voxels
    p.color_volume_dims = Vec3i::all(512);  //number of voxels
    p.semantic_volume_dims = Vec3i::all(512);  //number of voxels
    p.volume_size = Vec3f::all(3.f);  //meters
    p.volume_pose = Affine3f().translate(Vec3f(-p.volume_size[0]/2, -p.volume_size[1]/2, 0.5f));

    p.bilateral_sigma_depth = 0.04f;  //meter
    p.bilateral_sigma_spatial = 4.5; //pixels
    p.bilateral_kernel_size = 7;     //pixels

    p.icp_truncate_depth_dist = 0.f;        //meters, disabled
    p.icp_dist_thres = 0.25f;                //meters
    p.icp_angle_thres = deg2rad(20.f); //radians
    p.icp_iter_num.assign(iters, iters + levels);

    p.tsdf_min_camera_movement = 0.f; //meters, disabled
    p.tsdf_trunc_dist = 0.04f; //meters;
    p.tsdf_max_weight = 64;   //frames

    p.color_max_weight = 64;   //frames

    p.raycast_step_factor = 0.75f;  //in voxel sizes
    p.gradient_delta_factor = 0.5f; //in voxel sizes

    //p.light_pose = p.volume_pose.translation()/4; //meters
    p.light_pose = Vec3f::all(0.f); //meters

    return p;
}

kfusion::KinFu::KinFu(const KinFuParams& params) : frame_counter_(0), params_(params)
{
    CV_Assert(params.tsdf_volume_dims[0] % 32 == 0);
    CV_Assert(params.color_volume_dims[0] % 32 == 0);
    CV_Assert(params.semantic_volume_dims[0] % 32 == 0);

    tsdf_volume_ = cv::Ptr<cuda::TsdfVolume>(new cuda::TsdfVolume(params_.tsdf_volume_dims));

    tsdf_volume_->setTruncDist(params_.tsdf_trunc_dist);
    tsdf_volume_->setMaxWeight(params_.tsdf_max_weight);
    tsdf_volume_->setSize(params_.volume_size);
    tsdf_volume_->setPose(params_.volume_pose);
    tsdf_volume_->setRaycastStepFactor(params_.raycast_step_factor);
    tsdf_volume_->setGradientDeltaFactor(params_.gradient_delta_factor);

    if (params.integrate_color) {
        color_volume_ = cv::Ptr<cuda::ColorVolume>(new cuda::ColorVolume(params_.color_volume_dims));
        color_volume_->setTruncDist(params_.tsdf_trunc_dist);
        color_volume_->setMaxWeight(params_.color_max_weight);
        color_volume_->setSize(params_.volume_size);
        color_volume_->setPose(params_.volume_pose);
    }

    if (params.integrate_semantic) {
        semantic_volume_ = cv::Ptr<cuda::SemanticVolume>(new cuda::SemanticVolume(params_.semantic_volume_dims));
        semantic_volume_->setTruncDist(params_.tsdf_trunc_dist);
        semantic_volume_->setMaxWeight(params_.color_max_weight);
        semantic_volume_->setSize(params_.volume_size);
        semantic_volume_->setPose(params_.volume_pose);
    }

    icp_ = cv::Ptr<cuda::ProjectiveICP>(new cuda::ProjectiveICP());
    icp_->setDistThreshold(params_.icp_dist_thres);
    icp_->setAngleThreshold(params_.icp_angle_thres);
    icp_->setIterationsNum(params_.icp_iter_num);

    /*********************************************************************************
    * creating the optimization problem
    ********************************************************************************/

    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );

    keyframe_graph_.setAlgorithm(solver);
    keyframe_graph_.setVerbose(false);

    Eigen::Isometry3d p;
    p.setIdentity();
    VertexSE3* v = new VertexSE3();
    v->setId(vertex_id);
    v->setFixed(true);
    v->setEstimate(p);  
    v->setMarginalized(false);
    keyframe_graph_.addVertex(v);
    vec_keyframe_id.push_back(vertex_id);
    vertex_id++;
    previous_kf_vertex = v;
    pre_keyframe_idx = 0;

    allocate_buffers();
    reset();
}

const kfusion::KinFuParams& kfusion::KinFu::params() const
{ return params_; }

kfusion::KinFuParams& kfusion::KinFu::params()
{ return params_; }

const kfusion::cuda::TsdfVolume& kfusion::KinFu::tsdf() const
{ return *tsdf_volume_; }

kfusion::cuda::TsdfVolume& kfusion::KinFu::tsdf()
{ return *tsdf_volume_; }

const kfusion::cuda::ProjectiveICP& kfusion::KinFu::icp() const
{ return *icp_; }

const cv::Ptr<cuda::ColorVolume> kfusion::KinFu::color_volume () const
{
    if (params_.integrate_color)
        return color_volume_;
    return cv::Ptr<cuda::ColorVolume>();
}

cv::Ptr<cuda::ColorVolume> kfusion::KinFu::color_volume ()
{
    if (params_.integrate_color)
        return color_volume_;
    return cv::Ptr<cuda::ColorVolume>();
}

const cv::Ptr<cuda::SemanticVolume> kfusion::KinFu::semantic_volume () const
{
    if (params_.integrate_semantic)
        return semantic_volume_;
    return cv::Ptr<cuda::SemanticVolume>();
}

cv::Ptr<cuda::SemanticVolume> kfusion::KinFu::semantic_volume ()
{
    if (params_.integrate_semantic)
        return semantic_volume_;
    return cv::Ptr<cuda::SemanticVolume>();
}

kfusion::cuda::ProjectiveICP& kfusion::KinFu::icp()
{ return *icp_; }

void kfusion::KinFu::allocate_buffers()
{
    const int LEVELS = cuda::ProjectiveICP::MAX_PYRAMID_LEVELS;

    int cols = params_.cols;
    int rows = params_.rows;

    dists_.create(rows, cols);

    curr_.depth_pyr.resize(LEVELS);
    curr_.normals_pyr.resize(LEVELS);
    prev_.depth_pyr.resize(LEVELS);
    prev_.normals_pyr.resize(LEVELS);

    curr_.points_pyr.resize(LEVELS);
    prev_.points_pyr.resize(LEVELS);

    curr_.colors_pyr.resize(LEVELS);
    prev_.colors_pyr.resize(LEVELS);

    curr_.semantics_pyr.resize(LEVELS);
    prev_.semantics_pyr.resize(LEVELS);

    for(int i = 0; i < LEVELS; ++i)
    {
        curr_.depth_pyr[i].create(rows, cols);
        curr_.normals_pyr[i].create(rows, cols);

        prev_.depth_pyr[i].create(rows, cols);
        prev_.normals_pyr[i].create(rows, cols);

        curr_.points_pyr[i].create(rows, cols);
        prev_.points_pyr[i].create(rows, cols);

        curr_.colors_pyr[i].create(rows, cols);
        prev_.colors_pyr[i].create(rows, cols);

        curr_.semantics_pyr[i].create(rows, cols);
        prev_.semantics_pyr[i].create(rows, cols);

        cols /= 2;
        rows /= 2;
    }

    depths_.create(params_.rows, params_.cols);
    normals_.create(params_.rows, params_.cols);
    points_.create(params_.rows, params_.cols);
    colors_.create(params_.rows, params_.cols);
    semantics_.create(params_.rows, params_.cols);
}

void kfusion::KinFu::reset()
{
    if (frame_counter_)
        cout << "Reset" << endl;

    frame_counter_ = 0;
    poses_.clear();
    poses_.reserve(30000);
    poses_.push_back(Affine3f::Identity());
    tsdf_volume_->clear();
    if (params_.integrate_color)
        color_volume_->clear();
    if (params_.integrate_semantic)
    	semantic_volume_->clear();
}

kfusion::Affine3f kfusion::KinFu::getCameraPose (int time) const
{
    if (time > (int)poses_.size () || time < 0)
        time = (int)poses_.size () - 1;
    return poses_[time];
}

void kfusion::KinFu::writeKeyframePosesFromGraph(const std::string file_name)
{

    ofstream outfile(file_name);
    outfile << "#timestamp tx ty tz qx qy qz qw" << endl;
    for (int i = 0 ; i < vec_keyframe_id.size() ; i++)
//    for (int i = 0 ; i < vertex_id ; i++)
    {
        double temp[7] = {0,};
        keyframe_graph_.vertex(vec_keyframe_id[i])->getEstimateData(temp);
//        keyframe_graph_.vertex(i)->getEstimateData(temp);
        outfile << vec_timestamp[i] << " "
        		<< temp[0] << " "
                << temp[1] << " "
                << temp[2] << " "
                << temp[3] << " "
                << temp[4] << " "
                << temp[5] << " "
                << temp[6] << endl;
    }
    outfile.close();
}
void kfusion::KinFu::clearVolumes()
{
	tsdf_volume_->clear();
	if (params_.integrate_color)
		color_volume_->clear();
	if (params_.integrate_semantic)
		semantic_volume_->clear();
}

void kfusion::KinFu::redrawVolumes(const KinFuParams& p)
{
	// redraw
	for (int i = 0 ; i < vec_keyframe_id.size() ; i++)
	{
		double temp[7] = {0,};
		keyframe_graph_.vertex(vec_keyframe_id[i])->getEstimateData(temp);

		cv::Affine3f::Vec3 trans2(temp[0],temp[1],temp[2]);

		Eigen::Matrix3d rot1 = Quaternion(temp[6],temp[3],temp[4],temp[5]).toRotationMatrix();
		cv::Affine3f::Mat3 rot2(rot1(0,0),rot1(0,1),rot1(0,2),
								rot1(1,0),rot1(1,1),rot1(1,2),
								rot1(2,0),rot1(2,1),rot1(2,2));
		Affine3f pose(rot2,trans2);

		tsdf_volume_->integrate(vec_dist[i], pose, p.intr);
		if (p.integrate_color) {
			color_volume_->integrate(vec_image[i], vec_dist[i], pose, p.intr);
		}
		if (p.integrate_semantic) {
			semantic_volume_->integrate(vec_semantic[i], vec_dist[i], pose, p.intr);
		}
	}
}

void kfusion::KinFu::Affine3fToIsometry3d(const cv::Affine3f &from, Eigen::Isometry3d &to)
{
	Eigen::Matrix3d rot1;
	cv::Affine3f::Mat3 rot2 = from.rotation();
	rot1 << rot2(0,0),rot2(0,1),rot2(0,2),rot2(1,0),rot2(1,1),rot2(1,2),rot2(2,0),rot2(2,1),rot2(2,2);

	Eigen::Vector3d trans1;
	cv::Affine3f::Vec3 trans2 = from.translation();
	trans1 << trans2(0), trans2(1), trans2(2);

	to = rot1;
	to.translation() = trans1;
}

g2o::VertexSE3* kfusion::KinFu::addVertex(int vertex_id, Eigen::Isometry3d& current_pose)
{
//	cout << frame_counter_ << " " << vertex_id << endl;
	g2o::VertexSE3 *current_vertex = new VertexSE3();
	current_vertex->setEstimate(current_pose);
	current_vertex->setId(vertex_id);
	current_vertex->setMarginalized(false);
	keyframe_graph_.addVertex(current_vertex);

	return current_vertex;
}

void kfusion::KinFu::addEdge(int edge_id, g2o::VertexSE3* first_vertex, g2o::VertexSE3* second_vertex, Eigen::Isometry3d& constraint)
{
	EdgeSE3* e = new EdgeSE3();
	e->setId(edge_id);
	e->setMeasurement(constraint);
	e->resize(2);
	e->setVertex(0, first_vertex);
	e->setVertex(1, second_vertex);
	keyframe_graph_.addEdge(e);
}
void kfusion::KinFu::addEdge(int edge_id, g2o::OptimizableGraph::Vertex* first_vertex, g2o::VertexSE3* second_vertex, Eigen::Isometry3d& constraint)
{
	EdgeSE3* e = new EdgeSE3();
	e->setId(edge_id);
	e->setMeasurement(constraint);
	e->resize(2);
	e->setVertex(0, first_vertex);
	e->setVertex(1, second_vertex);
	keyframe_graph_.addEdge(e);
}
void kfusion::KinFu::addEdge(int edge_id, g2o::VertexSE3* first_vertex, g2o::OptimizableGraph::Vertex* second_vertex, Eigen::Isometry3d& constraint)
{
	EdgeSE3* e = new EdgeSE3();
	e->setId(edge_id);
	e->setMeasurement(constraint);
	e->resize(2);
	e->setVertex(0, first_vertex);
	e->setVertex(1, second_vertex);
	keyframe_graph_.addEdge(e);
}

bool kfusion::KinFu::estimateTransform(const cuda::Depth& source_depth, const cuda::Depth& target_depth, cv::Affine3f& transform, const int LEVELS, const KinFuParams& p)
{
	//source_frame
	 cuda::Frame source_frame;
	 {
		 source_frame.depth_pyr.resize(LEVELS);
		 source_frame.normals_pyr.resize(LEVELS);
		 source_frame.points_pyr.resize(LEVELS);

		 int cols = params_.cols;
		 int rows = params_.rows;
		 for(int i = 0; i < LEVELS; ++i)
		 {
			 source_frame.depth_pyr[i].create(rows, cols);
			 source_frame.normals_pyr[i].create(rows, cols);
			 source_frame.points_pyr[i].create(rows, cols);

			 cols /= 2;
			 rows /= 2;
		 }

		 cuda::depthBilateralFilter(source_depth, source_frame.depth_pyr[0], p.bilateral_kernel_size, p.bilateral_sigma_spatial, p.bilateral_sigma_depth);

		 if (p.icp_truncate_depth_dist > 0)
			 kfusion::cuda::depthTruncation(source_frame.depth_pyr[0], p.icp_truncate_depth_dist);

		 for (int i = 1; i < LEVELS; ++i)
			 cuda::depthBuildPyramid(source_frame.depth_pyr[i-1], source_frame.depth_pyr[i], p.bilateral_sigma_depth);

		 for (int i = 0; i < LEVELS; ++i)
			 cuda::computePointNormals(p.intr(i), source_frame.depth_pyr[i], source_frame.points_pyr[i], source_frame.normals_pyr[i]);
	 }
	 //target_frame
	 cuda::Frame target_frame;
	 {
		 target_frame.depth_pyr.resize(LEVELS);
		 target_frame.normals_pyr.resize(LEVELS);
		 target_frame.points_pyr.resize(LEVELS);

		 int cols = params_.cols;
		 int rows = params_.rows;
		 for(int i = 0; i < LEVELS; ++i)
		 {
			 target_frame.depth_pyr[i].create(rows, cols);
			 target_frame.normals_pyr[i].create(rows, cols);
			 target_frame.points_pyr[i].create(rows, cols);

			 cols /= 2;
			 rows /= 2;
		 }

		 cuda::depthBilateralFilter(target_depth, target_frame.depth_pyr[0], p.bilateral_kernel_size, p.bilateral_sigma_spatial, p.bilateral_sigma_depth);

		 if (p.icp_truncate_depth_dist > 0)
			 kfusion::cuda::depthTruncation(target_frame.depth_pyr[0], p.icp_truncate_depth_dist);

		 for (int i = 1; i < LEVELS; ++i)
			 cuda::depthBuildPyramid(target_frame.depth_pyr[i-1], target_frame.depth_pyr[i], p.bilateral_sigma_depth);

		 for (int i = 0; i < LEVELS; ++i)
			 cuda::computePointNormals(p.intr(i), target_frame.depth_pyr[i], target_frame.points_pyr[i], target_frame.normals_pyr[i]);
	 }


	 cv::Ptr<cuda::ProjectiveICP> icp_constraint;
	 icp_constraint = cv::Ptr<cuda::ProjectiveICP>(new cuda::ProjectiveICP());
	 icp_constraint->setDistThreshold(params_.icp_dist_thres);
	 icp_constraint->setAngleThreshold(params_.icp_angle_thres);
	 icp_constraint->setIterationsNum(params_.icp_iter_num);
	 bool ok = icp_constraint->estimateTransform(transform, p.intr, source_frame.points_pyr, source_frame.normals_pyr,
			 	 	 	 	 	 	 	 	 	 	 	 	 	    target_frame.points_pyr, target_frame.normals_pyr);

	 return ok;
}

void ConsolePrint(std::string color, std::string text)
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

bool kfusion::KinFu::operator()(const kfusion::cuda::Depth& depth, const kfusion::cuda::Image& image, const kfusion::cuda::Image& semantic, const std::string timestamp)
{
	bool keyframe_created = false;
	bool kfc_flag = false;
	int keyframe_interval = 5;
	int kfc_window = keyframe_interval/2+2;
	bool loop_closure_detected = false;

    const KinFuParams& p = params_;
    const int LEVELS = icp_->getUsedLevelsNum();

    cuda::computeDists(depth, dists_, p.intr);
    cuda::depthBilateralFilter(depth, curr_.depth_pyr[0], p.bilateral_kernel_size, p.bilateral_sigma_spatial, p.bilateral_sigma_depth);

    if (p.icp_truncate_depth_dist > 0)
        kfusion::cuda::depthTruncation(curr_.depth_pyr[0], p.icp_truncate_depth_dist);

    for (int i = 1; i < LEVELS; ++i)
        cuda::depthBuildPyramid(curr_.depth_pyr[i-1], curr_.depth_pyr[i], p.bilateral_sigma_depth);

    for (int i = 0; i < LEVELS; ++i)
        cuda::computePointNormals(p.intr(i), curr_.depth_pyr[i], curr_.points_pyr[i], curr_.normals_pyr[i]);

    if (frame_counter_ % keyframe_interval == 0)
    {
        cuda::Depth depth_copy; depth.copyTo(depth_copy);
        vec_depth.push_back(depth_copy);
        cuda::Dists dists_copy; dists_.copyTo(dists_copy);
    	vec_dist.push_back(dists_copy);
    	cuda::Image image_copy; image.copyTo(image_copy);
    	vec_image.push_back(image_copy);
    	cuda::Image semantic_copy; semantic.copyTo(semantic_copy);
    	vec_semantic.push_back(semantic_copy);
    	vec_timestamp.push_back(timestamp);
    }

    // sliding window
    if (sliding_vec_depth.size() > kfc_window){
    	sliding_vec_depth.erase(sliding_vec_depth.begin());
    }
    cuda::Depth depth_copy; depth.copyTo(depth_copy);
    sliding_vec_depth.push_back(depth_copy);

    cuda::waitAllDefaultStream();

    //can't perform more on first frame
    if (frame_counter_ == 0)
    {
        tsdf_volume_->integrate(dists_, poses_.back(), p.intr);
        curr_.points_pyr.swap(prev_.points_pyr);
        curr_.normals_pyr.swap(prev_.normals_pyr);
        curr_.colors_pyr.swap(prev_.colors_pyr);
        curr_.semantics_pyr.swap(prev_.semantics_pyr);
        return ++frame_counter_, false;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // ICP
    Affine3f affine;
    {
        //ScopeTime time("icp");
        bool ok = icp_->estimateTransform(affine, p.intr, curr_.points_pyr, curr_.normals_pyr, prev_.points_pyr, prev_.normals_pyr);

        if (!ok) // failure case
        {
            return reset(), false;
        }
    }
    poses_.push_back(poses_.back() * affine);

    // keyframe graph construction
    if (frame_counter_ % keyframe_interval == 0) {
    	cur_keyframe_idx = frame_counter_;

    	ConsolePrint("green", "[keyframe created] ");
    	cout << "keyframe idx: " << cur_keyframe_idx << endl;

    	// transformation between the previous keyframe and the current keyframe
    	cv::Affine3f keyframe_odom = cv::Affine3f::Identity();
    	keyframe_odom = poses_[pre_keyframe_idx].inv()*poses_[cur_keyframe_idx];

		Eigen::Isometry3d edge_con;
		Affine3fToIsometry3d(keyframe_odom, edge_con);

		// current pose
		Eigen::Isometry3d current_pose;
		Affine3fToIsometry3d(poses_.back(),current_pose);

//		cout << poses_.back().translation() << endl;

		current_kf_vertex = addVertex(vertex_id, current_pose);
		addEdge(edge_id, previous_kf_vertex, current_kf_vertex, edge_con);

		vec_keyframe_id.push_back(vertex_id);
		vertex_id++;edge_id++;

		// backward_kfc
		if (kfc_flag)
		{
			for (int i = 0 ; i < kfc_window ; i++)
			{
				int target_keyframe_idx = vec_keyframe_id.back();
				int current_vertex_idx = vec_keyframe_id.back()-kfc_window+i;

				ConsolePrint("red", "[backward KFC] ");
				cout << "target keyframe idx: " << target_keyframe_idx << " "
						"current vertex idx " << current_vertex_idx << endl;

				// KFC from ICP
				Affine3f backward_kfc_transform;
				cuda::Depth current_depth = sliding_vec_depth[i];
				cuda::Depth previous_keyframe_depth = vec_depth.back();
				estimateTransform(previous_keyframe_depth, current_depth, backward_kfc_transform, LEVELS, p);// original
				cout << "KFC from ICP " << backward_kfc_transform.translation() << endl;

				// KFC from pose vector
//				backward_kfc_transform = poses_[current_vertex_idx].inv()*poses_[target_keyframe_idx];
//				cout << "KFC from pose vector " << backward_kfc_transform.translation() << endl;

				Eigen::Isometry3d backward_kfc_edge;
				Affine3fToIsometry3d(backward_kfc_transform, backward_kfc_edge);

				// current vertex position
				Eigen::Isometry3d current_pose;
				Affine3fToIsometry3d(poses_[current_vertex_idx],current_pose);

				//cout << poses_[current_vertex_idx].translation() << endl;

				g2o::OptimizableGraph::Vertex* current_vertex = keyframe_graph_.vertex(current_vertex_idx);
				addEdge(edge_id, current_vertex, previous_kf_vertex, backward_kfc_edge);

				edge_id++;
			}
		}

		previous_kf_vertex = current_kf_vertex;
		pre_keyframe_idx = cur_keyframe_idx;

//		keyframe_created = true;
    }
    // forward_kfc, with edge
    else if (frame_counter_ - pre_keyframe_idx <= kfc_window && kfc_flag){
    	int target_keyframe_idx = pre_keyframe_idx;
    	int current_vertex_idx = vertex_id;

    	ConsolePrint("yellow", "[forward KFC] ");
    	cout << "target keyframe idx: " << target_keyframe_idx << " "
    			"current vertex idx " << current_vertex_idx << endl;

    	// KFC from ICP
    	Affine3f forward_kfc_transform;
		cuda::Depth current_depth = depth;
		cuda::Depth previous_keyframe_depth = vec_depth.back();
    	estimateTransform(previous_keyframe_depth, current_depth, forward_kfc_transform, LEVELS, p);
    	cout << "KFC from ICP " << forward_kfc_transform.translation() << endl;

    	// KFC from pose vector
//    	forward_kfc_transform = poses_[current_vertex_idx].inv()*poses_[target_keyframe_idx];
//    	cout << "KFC from pose vector " << forward_kfc_transform.translation() << endl;

		Eigen::Isometry3d forward_kfc_edge;
		Affine3fToIsometry3d(forward_kfc_transform,forward_kfc_edge);

		// current vertex position
		Eigen::Isometry3d current_pose;
		Affine3fToIsometry3d(poses_.back(),current_pose);

		// cout << poses_.back().translation() << endl;

		g2o::VertexSE3 *current_vertex = addVertex(current_vertex_idx, current_pose);
		addEdge(edge_id, current_vertex, previous_kf_vertex, forward_kfc_edge);

		vertex_id++; edge_id++;
    }
    // forward kfc, vertex only
    else
    {
		// current vertex position
		Eigen::Isometry3d current_pose;
		Affine3fToIsometry3d(poses_.back(),current_pose);

		g2o::VertexSE3 *current_vertex = addVertex(vertex_id, current_pose);

		vertex_id++;
    }

    // loop closure detection
//    int loop_closure_idx1 = 0;  // be careful!
//    int loop_closure_idx2 = 100;
//    if (frame_counter_ == loop_closure_idx2) {
//		 cout << "[Loop closure is detected.]" << endl;
//
//		 Affine3f lcc_transform;
//		 estimateTransform(depth,vec_depth[loop_closure_idx1/keyframe_interval],lcc_transform, LEVELS, p);
//
//		 // cout << loop_closure_transform.translation() << endl;
//
//		 Eigen::Isometry3d lc_con;
//		 Affine3fToIsometry3d(lcc_transform,lc_con);
//		 addEdge(edge_id, keyframe_graph_.vertex(loop_closure_idx1/keyframe_interval), current_kf_vertex, lc_con);
//
//		 edge_id++;
//         loop_closure_detected = true;
//    }

    // graph optimization
	if (loop_closure_detected || keyframe_created) {
		cout << endl << "[Graph optimization is triggered.]" << endl;

		// incremental PGO
//		for (int i = 0 ; i < cur_keyframe_idx-2*keyframe_interval ; i++)
//		{
//			keyframe_graph_.vertex(i)->setFixed(true);
//		}

		keyframe_graph_.setVerbose(true);
		keyframe_graph_.initializeOptimization();
		keyframe_graph_.computeInitialGuess();
		keyframe_graph_.optimize(2);

		clearVolumes();
		redrawVolumes(p);

		if (kfc_flag)
			writeKeyframePosesFromGraph("poses_with_KFC.txt");
		else
			writeKeyframePosesFromGraph("poses_without_KFC.txt");

        loop_closure_detected = false;
        keyframe_created = false;
		return ++frame_counter_, true;
	}

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Volume integration

    // We do not integrate volume if camera does not move.
    float rnorm = (float)cv::norm(affine.rvec());
    float tnorm = (float)cv::norm(affine.translation());
    bool integrate = (rnorm + tnorm)/2 >= p.tsdf_min_camera_movement;
    if (integrate)
    {
        //ScopeTime time("tsdf");
        tsdf_volume_->integrate(dists_, poses_.back(), p.intr);
        if (p.integrate_color) {
            color_volume_->integrate(image, dists_, poses_.back(), p.intr);
        }
        if (p.integrate_semantic) {
            semantic_volume_->integrate(semantic, dists_, poses_.back(), p.intr);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Ray casting
    {
        //ScopeTime time("ray-cast-all");
        tsdf_volume_->raycast(poses_.back(), p.intr, prev_.points_pyr[0], prev_.normals_pyr[0], 
                                prev_.colors_pyr[0], *color_volume_,
                                prev_.semantics_pyr[0], *semantic_volume_);

        for (int i = 1; i < LEVELS; ++i)
            resizePointsNormals(prev_.points_pyr[i-1], prev_.normals_pyr[i-1], prev_.points_pyr[i], prev_.normals_pyr[i]);
        cuda::waitAllDefaultStream();
    }

    return ++frame_counter_, true;
}

void kfusion::KinFu::renderImage(cuda::Image& image, int flag)
{
    const KinFuParams& p = params_;
    image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);

    if (flag < 1 || flag > 3)
        cuda::renderImage(prev_.points_pyr[0], prev_.normals_pyr[0], params_.intr, params_.light_pose, image);
    else if (flag == 2)
        cuda::renderTangentColors(prev_.normals_pyr[0], image);
    else /* if (flag == 3) */
    {
        DeviceArray2D<RGB> i1(p.rows, p.cols, image.ptr(), image.step());
        DeviceArray2D<RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());

        cuda::renderImage(prev_.points_pyr[0], prev_.normals_pyr[0], params_.intr, params_.light_pose, i1);
        cuda::renderTangentColors(prev_.normals_pyr[0], i2);

        // raycasted color 
        cv::Mat color_host(p.rows, p.cols, CV_8UC4);
        prev_.colors_pyr[0].download(color_host.ptr<RGB>(), params_.cols*4);
        cv::Mat color_host2(p.rows, p.cols, CV_8UC3);
        cvtColor(color_host, color_host2, CV_BGRA2BGR);
        // cv::imshow("raycasted color", color_host2);

        // raycasted semantic
        cv::Mat semantic_host(p.rows, p.cols, CV_8UC4);
        prev_.semantics_pyr[0].download(semantic_host.ptr<RGB>(), params_.cols*4);
        cv::Mat semantic_host2(p.rows, p.cols, CV_8UC3);
        cvtColor(semantic_host, semantic_host2, CV_BGRA2BGR);
        // cv::imshow("raycasted semantic", semantic_host2);

        cv::Mat concat;
        cv::hconcat(color_host2, semantic_host2, concat);
        cv::imshow("raycasted color / semantic", concat);
    }
}


void kfusion::KinFu::renderImage(cuda::Image& image, const Affine3f& pose, int flag)
{
    const KinFuParams& p = params_;
    image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);
    depths_.create(p.rows, p.cols);
    normals_.create(p.rows, p.cols);
    points_.create(p.rows, p.cols);
    colors_.create(p.rows, p.cols);
    semantics_.create(p.rows, p.cols);

    tsdf_volume_->raycast(pose, p.intr, points_, normals_, colors_, *color_volume_, semantics_, *semantic_volume_);

    if (flag < 1 || flag > 3)
        cuda::renderImage(points_, normals_, params_.intr, params_.light_pose, image);
    else if (flag == 2)
        cuda::renderTangentColors(normals_, image);
    else /* if (flag == 3) */
    {
        DeviceArray2D<RGB> i1(p.rows, p.cols, image.ptr(), image.step());
        DeviceArray2D<RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());

        cuda::renderImage(points_, normals_, params_.intr, params_.light_pose, i1);
        cuda::renderTangentColors(normals_, i2);

        // raycasted color 
        cv::Mat color_host(p.rows, p.cols, CV_8UC4);
        colors_.download(color_host.ptr<RGB>(), params_.cols*4);
        cv::Mat color_host2(p.rows, p.cols, CV_8UC3);
        cvtColor(color_host, color_host2, CV_BGRA2BGR);
        // cv::imshow("raycasted color", color_host2);

        // raycasted semantic
        cv::Mat semantic_host(p.rows, p.cols, CV_8UC4);
        semantics_.download(semantic_host.ptr<RGB>(), params_.cols*4);
        cv::Mat semantic_host2(p.rows, p.cols, CV_8UC3);
        cvtColor(semantic_host, semantic_host2, CV_BGRA2BGR);
        // cv::imshow("raycasted semantic", semantic_host2);

        cv::Mat concat;
        cv::hconcat(color_host2, semantic_host2, concat);
        cv::imshow("raycasted color / semantic", concat);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//namespace pcl
//{
//    Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix)
//    {
//        Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
//        Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

//        double rx = R(2, 1) - R(1, 2);
//        double ry = R(0, 2) - R(2, 0);
//        double rz = R(1, 0) - R(0, 1);

//        double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
//        double c = (R.trace() - 1) * 0.5;
//        c = c > 1. ? 1. : c < -1. ? -1. : c;

//        double theta = acos(c);

//        if( s < 1e-5 )
//        {
//            double t;

//            if( c > 0 )
//                rx = ry = rz = 0;
//            else
//            {
//                t = (R(0, 0) + 1)*0.5;
//                rx = sqrt( std::max(t, 0.0) );
//                t = (R(1, 1) + 1)*0.5;
//                ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
//                t = (R(2, 2) + 1)*0.5;
//                rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

//                if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
//                    rz = -rz;
//                theta /= sqrt(rx*rx + ry*ry + rz*rz);
//                rx *= theta;
//                ry *= theta;
//                rz *= theta;
//            }
//        }
//        else
//        {
//            double vth = 1/(2*s);
//            vth *= theta;
//            rx *= vth; ry *= vth; rz *= vth;
//        }
//        return Eigen::Vector3d(rx, ry, rz).cast<float>();
//    }
//}


