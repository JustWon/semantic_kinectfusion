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

    graph_.setAlgorithm(solver);
    graph_.setVerbose(false);

    Eigen::Isometry3d p;
    p.setIdentity();
    VertexSE3* v = new VertexSE3();
    v->setId(vertex_id);
    v->setFixed(true);
    v->setEstimate(p);  
    v->setMarginalized(false);
    graph_.addVertex(v);
    vertex_id++;
    previous_vertex = v;
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

bool kfusion::KinFu::operator()(const kfusion::cuda::Depth& depth, const kfusion::cuda::Image& image, const kfusion::cuda::Image& semantic)
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

    cuda::Dists dists_copy; dists_.copyTo(dists_copy);
	vec_dist.push_back(dists_copy);
	cuda::Image image_copy; image.copyTo(image_copy);
	vec_image.push_back(image_copy);
	cuda::Image semantic_copy; semantic.copyTo(semantic_copy);
	vec_semantic.push_back(semantic_copy);

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
    Affine3f affine; // cuur -> prev
    {
        //ScopeTime time("icp");
        bool ok = icp_->estimateTransform(affine, p.intr, curr_.points_pyr, curr_.normals_pyr, prev_.points_pyr, prev_.normals_pyr);

        if (!ok)
            return reset(), false;
    }

    poses_.push_back(poses_.back() * affine); // curr -> global

    // graph construction
    int keyframe_interval = 10;
    if (frame_counter_ % keyframe_interval == 0) {
    	cur_keyframe_idx = frame_counter_;

    	// transformation between the previous keyframe and the current keyframe
    	cv::Affine3f keyframe_odom = cv::Affine3f::Identity();
    	keyframe_odom = poses_[pre_keyframe_idx].inv()*poses_[cur_keyframe_idx];

		Eigen::Isometry3d edge_con;

		Eigen::Matrix3d rot1;
		cv::Affine3f::Mat3 rot2 = keyframe_odom.rotation();
		rot1 << rot2(0,0),rot2(0,1),rot2(0,2),rot2(1,0),rot2(1,1),rot2(1,2),rot2(2,0),rot2(2,1),rot2(2,2);

		Eigen::Vector3d trans1;
		cv::Affine3f::Vec3 trans2 = keyframe_odom.translation();
		trans1 << trans2(0), trans2(1), trans2(2);

		edge_con = rot1;
		edge_con.translation() = trans1;

		// current pose
		Eigen::Isometry3d current_pose;

		Eigen::Matrix3d cur_pose_rot1;
		cv::Affine3f::Mat3 cur_pose_rot2 = poses_.back().rotation();
		cur_pose_rot1 << cur_pose_rot2(0,0),cur_pose_rot2(0,1),cur_pose_rot2(0,2),
						 cur_pose_rot2(1,0),cur_pose_rot2(1,1),cur_pose_rot2(1,2),
						 cur_pose_rot2(2,0),cur_pose_rot2(2,1),cur_pose_rot2(2,2);

		Eigen::Vector3d cur_pose_trans1;
		cv::Affine3f::Vec3 cur_pose_trans2 = poses_.back().translation();
		cur_pose_trans1 << cur_pose_trans2(0), cur_pose_trans2(1), cur_pose_trans2(2);

		current_pose = cur_pose_rot1;
		current_pose.translation() = cur_pose_trans1;

		current_vertex = new VertexSE3();
		current_vertex->setEstimate(current_pose);
		current_vertex->setId(vertex_id);
		current_vertex->setMarginalized(false);
		graph_.addVertex(current_vertex);

		EdgeSE3* e = new EdgeSE3();
		e->setId(edge_id);
		e->setMeasurement(edge_con);
		e->resize(2);
		e->setVertex(0, previous_vertex);
		e->setVertex(1, current_vertex);
		graph_.addEdge(e);

		previous_vertex = current_vertex;
		pre_keyframe_idx = cur_keyframe_idx;

		vertex_id++;edge_id++;
    }

    // graph optimization
	if (frame_counter_ % 100 == 0) {
		cout << "[Graph optimization is triggered.]" << endl;

		graph_.initializeOptimization();
		graph_.computeInitialGuess();
		graph_.optimize(10);

		// clear
		{
			tsdf_volume_->clear();
			if (params_.integrate_color)
				color_volume_->clear();
			if (params_.integrate_semantic)
				semantic_volume_->clear();
		}

		// redraw
		for (int i = 0 ; i < vertex_id ; i++)
		{
			double temp[7] = {0,};
			graph_.vertex(i)->getEstimateData(temp);

			cv::Affine3f::Vec3 trans2(temp[0],temp[1],temp[2]);

			Eigen::Matrix3d rot1 = Quaternion(temp[6],temp[3],temp[4],temp[5]).toRotationMatrix();
			cv::Affine3f::Mat3 rot2(rot1(0,0),rot1(0,1),rot1(0,2),
									rot1(1,0),rot1(1,1),rot1(1,2),
									rot1(2,0),rot1(2,1),rot1(2,2));
			Affine3f pose(rot2,trans2);

			tsdf_volume_->integrate(vec_dist[keyframe_interval*i], pose, p.intr);
			if (p.integrate_color) {
				color_volume_->integrate(vec_image[keyframe_interval*i], vec_dist[keyframe_interval*i], pose, p.intr);
			}
			if (p.integrate_semantic) {
				semantic_volume_->integrate(vec_semantic[keyframe_interval*i], vec_dist[keyframe_interval*i], pose, p.intr);
			}
		}
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


