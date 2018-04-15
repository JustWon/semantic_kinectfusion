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

#include <DW_Utility.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <fstream>

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
    p.integrate_color = true;
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

    allocate_buffers();
    reset();

    /*********************************************************************************
    * creating the optimization problem
    ********************************************************************************/

    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );

    pose_graph.setAlgorithm(solver);
    pose_graph.setVerbose(false);
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

    // pose graph clear & init
    pose_graph.clear();
    vertex_id = 0;
    edge_id =0;
    Eigen::Isometry3d p;
    p.setIdentity();
    VertexSE3* v = new VertexSE3();
    v->setId(vertex_id);
    v->setFixed(true);
    v->setEstimate(p);
    v->setMarginalized(false);
    pose_graph.addVertex(v);
    vec_keyframe_id.push_back(vertex_id);
    vertex_id++;
    previous_kf_vertex = v;
    previous_vertex = v;
    pre_keyframe_idx = 0;
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
        pose_graph.vertex(vec_keyframe_id[i])->getEstimateData(temp);
//        pose_graph.vertex(i)->getEstimateData(temp);
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
		pose_graph.vertex(vec_keyframe_id[i])->getEstimateData(temp);

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
	pose_graph.addVertex(current_vertex);

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
	pose_graph.addEdge(e);
}
void kfusion::KinFu::addEdge(int edge_id, g2o::OptimizableGraph::Vertex* first_vertex, g2o::VertexSE3* second_vertex, Eigen::Isometry3d& constraint)
{
	EdgeSE3* e = new EdgeSE3();
	e->setId(edge_id);
	e->setMeasurement(constraint);
	e->resize(2);
	e->setVertex(0, first_vertex);
	e->setVertex(1, second_vertex);
	pose_graph.addEdge(e);
}
void kfusion::KinFu::addEdge(int edge_id, g2o::VertexSE3* first_vertex, g2o::OptimizableGraph::Vertex* second_vertex, Eigen::Isometry3d& constraint)
{
	EdgeSE3* e = new EdgeSE3();
	e->setId(edge_id);
	e->setMeasurement(constraint);
	e->resize(2);
	e->setVertex(0, first_vertex);
	e->setVertex(1, second_vertex);
	pose_graph.addEdge(e);
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

//subvolume approach
void kfusion::KinFu::storeSubvolume()
{
	cuda::DeviceArray<Point> cloud_buffer_;
	cuda::DeviceArray<RGB> color_buffer_;

	cuda::DeviceArray<Point> cloud = this->tsdf().fetchCloud(cloud_buffer_);
	cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);

	this->color_volume()->fetchColors(cloud, color_buffer_);
	cv::Mat color_host(1, (int)cloud.size(), CV_8UC4);
	cloud.download(cloud_host.ptr<Point>());
	color_buffer_.download(color_host.ptr<RGB>());

	// To do: it should be a tsdf volume rather than a point cloud.
	vec_subvolume.push_back(cloud_host);

	// if several subvolumes exist in the vector, register them
	// how to register them..?



//	for (int i = 0 ; i < vec_subvolume.size() ; i++)
//		cout << "vec_subvolume[" << i << "].cols : " << vec_subvolume[i].cols << endl;
}

void kfusion::KinFu::storePoseVector()
{
	vec_poses.push_back(this->poses_);
}

std::string padding(std::string value)
{
	return "padding" + value + "padding";
}

std::string padding(double value)
{
    char temp[100];
    sprintf(temp, "%0.10f", value);
    return "padding" + string(temp) + "padding";
}

void kfusion::KinFu::savePoseVector(std::string output_filename)
{
    DW_Utility::consolePrint("[savePoseVector] \n", "green");

    // sort(pose_graph.edges().begin(), pose_graph.edges().end(), IncrementalEdgesCompare());

    using boost::property_tree::ptree;

    ptree posegraph;
    ptree edges, nodes;

    posegraph.put("class_name", "PoseGraph");

    vector<SparseOptimizer::Edge*> edgesv;
    for (SparseOptimizer::EdgeSet::iterator it = pose_graph.edges().begin(); it != pose_graph.edges().end(); ++it) {
      SparseOptimizer::Edge* e = dynamic_cast<SparseOptimizer::Edge*>(*it);
      edgesv.push_back(e);
    }

    // sort the edges in a way that inserting them makes sense
    sort(edgesv.begin(), edgesv.end(), SparseOptimizer::EdgeIDCompare());

    for (vector<SparseOptimizer::Edge*>::iterator it = edgesv.begin(); it != edgesv.end(); ++it) {
        SparseOptimizer::Edge* e = *it;

        double meas[10] = {0,};
        e->getMeasurementData(meas);
        cv::Affine3f::Vec3 trans(meas[0],meas[1],meas[2]);
        Eigen::Matrix3d rot = Quaternion(meas[6],meas[3],meas[4],meas[5]).toRotationMatrix();

        double* information = e->informationData();

        ptree posegraphedge;
        ptree information_node, transformation_node;
        ptree value;

        {
            // rot(0);-rot(1);-rot(2);trans(0);
            // -rot(3);rot(4);-rot(5);trans(1);
            // -rot(6);-rot(7);rot(8);trans(2);
            // 0; 0; 0; 1

            value.put("", padding((rot(0))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((rot(3))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((rot(6))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((0.0)));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((rot(1))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((rot(4))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((rot(7))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((0.0)));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((rot(2))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((rot(5))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((rot(8))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((0.0)));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((trans(0))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((trans(1))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((trans(2))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((1.0)));
            transformation_node.push_back(std::make_pair("", value));
            
        }

        for (int i = 0 ; i < 36 ;i++){
            value.put("", padding(to_string(information[i])));
            information_node.push_back(std::make_pair("", value));
        }

        posegraphedge.put("class_name", "PoseGraphEdge");
        posegraphedge.put("confidence", padding(to_string(1.0))); // something wrong
        posegraphedge.add_child("information", information_node);
        posegraphedge.put("source_node_id", padding(to_string(e->vertices()[0]->id())));
        posegraphedge.put("target_node_id", padding(to_string(e->vertices()[1]->id())));
        posegraphedge.add_child("transformation", transformation_node);
        posegraphedge.put("uncertain", padding("false"));
        posegraphedge.put("version_major", padding(to_string(1)));
        posegraphedge.put("version_minor", padding(to_string(0)));

        edges.push_back(std::make_pair("", posegraphedge));
    }
    posegraph.add_child("edges", edges);


    for (int i = 0 ; i < 100 ; i++)
    {
        Eigen::Isometry3d node_pose;
        Affine3fToIsometry3d(poses_[i],node_pose);

        // cout << node_pose.rotation() << endl;
        // cout << node_pose.translation() << endl;

        ptree posegraphnode;
        ptree transformation_node;
        ptree value;

        {
            // rot(0);rot(1);rot(2);trans(0);
            // rot(3);rot(4);rot(5);trans(1);
            // rot(6);rot(7);rot(8);trans(2);
            // 0; 0; 0; 1

            // cout << node_pose.rotation()(6) << endl;
            // cout << "to_string(node_pose.rotation()(6))" << to_string(node_pose.rotation()(6)) << endl;
            // cout << "padding(node_pose.rotation()(6))" << padding(node_pose.rotation()(6)) << endl;


            value.put("", padding((node_pose.rotation()(0))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((node_pose.rotation()(1))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((node_pose.rotation()(2))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((0.0)));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((node_pose.rotation()(3))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((node_pose.rotation()(4))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((node_pose.rotation()(5))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((0.0)));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((node_pose.rotation()(6))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((node_pose.rotation()(7))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((node_pose.rotation()(8))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((0.0)));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((node_pose.translation()(0))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((node_pose.translation()(1))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((node_pose.translation()(2))));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding((1.0)));
            transformation_node.push_back(std::make_pair("", value));
            
        }

        posegraphnode.put("class_name", "PoseGraphNode");
        posegraphnode.add_child("pose", transformation_node);
        posegraphnode.put("version_major", padding(to_string(1)));
        posegraphnode.put("version_minor", padding(to_string(0)));

        nodes.push_back(std::make_pair("", posegraphnode));
    }
    posegraph.add_child("nodes", nodes);
    posegraph.put("version_major", padding(to_string(1)));
    posegraph.put("version_minor", padding(to_string(0)));

    write_json(output_filename, posegraph);

    //json post-processing
    ifstream filein(output_filename);
    ofstream fileout(output_filename+"temp");
    string strTemp;
    while (getline(filein, strTemp))
    {
        boost::replace_all(strTemp, "\"padding", "");
        boost::replace_all(strTemp, "padding\"", "");
        fileout << strTemp << endl;
    }
    std::remove(output_filename.c_str());
    std::rename((output_filename+"temp").c_str(), output_filename.c_str());

}


void kfusion::KinFu::savePoseGraph(std::string output_filename)
{
	DW_Utility::consolePrint("[savePoseGraph] \n", "green");

	using boost::property_tree::ptree;

	ptree posegraph;
	ptree edges, nodes;

	posegraph.put("class_name", "PoseGraph");

	vector<SparseOptimizer::Edge*> edgesv;
    for (SparseOptimizer::EdgeSet::iterator it = pose_graph.edges().begin(); it != pose_graph.edges().end(); ++it) {
      SparseOptimizer::Edge* e = dynamic_cast<SparseOptimizer::Edge*>(*it);
      edgesv.push_back(e);
    }

    // sort the edges in a way that inserting them makes sense
    sort(edgesv.begin(), edgesv.end(), SparseOptimizer::EdgeIDCompare());

	for (vector<SparseOptimizer::Edge*>::iterator it = edgesv.begin(); it != edgesv.end(); ++it) {
      	SparseOptimizer::Edge* e = *it;

		double meas[10] = {0,};
		e->getMeasurementData(meas);
		cv::Affine3f::Vec3 trans(meas[0],meas[1],meas[2]);
		Eigen::Matrix3d rot = Quaternion(meas[6],meas[3],meas[4],meas[5]).toRotationMatrix();

		double* information = e->informationData();

		ptree posegraphedge;
		ptree information_node, transformation_node;
		ptree value;

		{
			// rot(0);-rot(1);-rot(2);trans(0);
			// -rot(3);rot(4);-rot(5);trans(1);
			// -rot(6);-rot(7);rot(8);trans(2);
			// 0; 0; 0; 1

			value.put("", padding((rot(0))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(3))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(6))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((0.0)));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(1))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(4))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(7))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((0.0)));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(2))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(5))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(8))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((0.0)));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((trans(0))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((trans(1))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((trans(2))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((1.0)));
			transformation_node.push_back(std::make_pair("", value));
			
		}

		for (int i = 0 ; i < 36 ;i++){
			value.put("", padding(to_string(information[i])));
			information_node.push_back(std::make_pair("", value));
		}

		posegraphedge.put("class_name", "PoseGraphEdge");
		posegraphedge.put("confidence", padding(to_string(1.0))); // something wrong
		posegraphedge.add_child("information", information_node);
		posegraphedge.put("source_node_id", padding(to_string(e->vertices()[0]->id())));
		posegraphedge.put("target_node_id", padding(to_string(e->vertices()[1]->id())));
		posegraphedge.add_child("transformation", transformation_node);
		posegraphedge.put("uncertain", padding("false"));
		posegraphedge.put("version_major", padding(to_string(1)));
		posegraphedge.put("version_minor", padding(to_string(0)));

		edges.push_back(std::make_pair("", posegraphedge));
	}
	posegraph.add_child("edges", edges);


	for (int i = 0 ; i < 100 ; i++)
	{
		double esti[10] = {0,};
		pose_graph.vertex(i)->getEstimateData(esti);
		cv::Affine3f::Vec3 trans(esti[0],esti[1],esti[2]);
		Eigen::Matrix3d rot = Quaternion(esti[6],esti[3],esti[4],esti[5]).toRotationMatrix().inverse();


        Eigen::Isometry3d node_pose;
        Affine3fToIsometry3d(poses_[i],node_pose);

		ptree posegraphnode;
		ptree transformation_node;
		ptree value;

		{
            // rot(0);rot(1);rot(2);trans(0);
            // rot(3);rot(4);rot(5);trans(1);
            // rot(6);rot(7);rot(8);trans(2);
            // 0; 0; 0; 1

			value.put("", padding((rot(0))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(3))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(6))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((0.0)));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(1))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(4))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(7))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((0.0)));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(2))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(5))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((rot(8))));
			transformation_node.push_back(std::make_pair("", value));
			value.put("", padding((0.0)));
			transformation_node.push_back(std::make_pair("", value));


			// value.put("", padding((node_pose.translation()(0))));
			// transformation_node.push_back(std::make_pair("", value));
			// value.put("", padding((node_pose.translation()(1))));
			// transformation_node.push_back(std::make_pair("", value));
			// value.put("", padding((node_pose.translation()(2))));
			// transformation_node.push_back(std::make_pair("", value));
			// value.put("", padding((1.0)));
			// transformation_node.push_back(std::make_pair("", value));

            value.put("", padding(trans(0)));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding(trans(1)));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding(trans(2)));
            transformation_node.push_back(std::make_pair("", value));
            value.put("", padding(1.0));
            transformation_node.push_back(std::make_pair("", value));
		}

		posegraphnode.put("class_name", "PoseGraphNode");
		posegraphnode.add_child("pose", transformation_node);
		posegraphnode.put("version_major", padding(to_string(1)));
		posegraphnode.put("version_minor", padding(to_string(0)));

		nodes.push_back(std::make_pair("", posegraphnode));
	}
	posegraph.add_child("nodes", nodes);
	posegraph.put("version_major", padding(to_string(1)));
	posegraph.put("version_minor", padding(to_string(0)));

	write_json(output_filename, posegraph);

	//json post-processing
	ifstream filein(output_filename);
	ofstream fileout(output_filename+"temp");
	string strTemp;
	while (getline(filein, strTemp))
	{
		boost::replace_all(strTemp, "\"padding", "");
		boost::replace_all(strTemp, "padding\"", "");
		fileout << strTemp << endl;
	}
	std::remove(output_filename.c_str());
	std::rename((output_filename+"temp").c_str(), output_filename.c_str());

}

void kfusion::KinFu::saveEstimatedTrajectories()
{
	DW_Utility::consolePrint("saveEstimatedTrajectories", "green");
	storePoseVector();

	ofstream ofile("estimated_trajectories.txt");
	ofile << "#timestamp tx ty tz qx qy qz qw" << endl;

	int temp_idx = 0;
	for (int i = 0 ; i < vec_poses.size() ; i++)
	{
		std::vector<Affine3f> cur_poses = vec_poses[i];
		for (int j = 0 ; j < cur_poses.size() ; j++)
		{
			ofile << vec_timestamp[temp_idx++] << " "
				  << cur_poses[j].translation()[0] << " "
				  << cur_poses[j].translation()[1] << " "
				  << cur_poses[j].translation()[2] << " "
				  << "0 0 0 1" << endl;
		}
	}
	ofile.close();
}

Affine3f kfusion::KinFu::getLastSucessPose()
{
	Affine3f ret;
	if (vec_poses.size() > 0)
		ret = vec_poses.back().back();
	return ret;
}

int kfusion::KinFu::operator()(const kfusion::cuda::Depth& depth, const kfusion::cuda::Image& image, const kfusion::cuda::Image& semantic, const std::string timestamp)
{
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


    vec_timestamp.push_back(timestamp);


    cuda::waitAllDefaultStream();

    //can't perform more on first frame
    if (frame_counter_ == 0)
    {
        tsdf_volume_->integrate(dists_, poses_.back(), p.intr);
        curr_.points_pyr.swap(prev_.points_pyr);
        curr_.normals_pyr.swap(prev_.normals_pyr);
        curr_.colors_pyr.swap(prev_.colors_pyr);
        curr_.semantics_pyr.swap(prev_.semantics_pyr);
        return ++frame_counter_, 0;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // ICP
    Affine3f affine;
    {
        //ScopeTime time("icp");
        bool ok = icp_->estimateTransform(affine, p.intr, curr_.points_pyr, curr_.normals_pyr, prev_.points_pyr, prev_.normals_pyr);

        if (!ok) // failure case
        {
        	cout << "tracking failure" << endl;
            return 2;
        }
    }
    poses_.push_back(poses_.back() * affine);

    ///////////////////////////////////////////////////////////////////////////////////////////
    // graph construction
    {
		// current vertex position
		Eigen::Isometry3d edge_odom;
		Affine3fToIsometry3d(affine,edge_odom);

		Eigen::Isometry3d node_pose;
        Affine3fToIsometry3d(poses_.back(),node_pose);

        // cout << setprecision(10);
        // cout << node_pose.rotation() << endl;
        // cout << node_pose.translation() << endl;

		g2o::VertexSE3 *current_vertex = addVertex(vertex_id, node_pose);
		addEdge(edge_id, previous_vertex, current_vertex, edge_odom);

		previous_vertex = current_vertex;
		vertex_id++;edge_id++;
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

    return ++frame_counter_, 1;
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


