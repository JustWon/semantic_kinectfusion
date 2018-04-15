#pragma once

#include <kfusion/types.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/cuda/color_volume.hpp>
#include <kfusion/cuda/semantic_volume.hpp>
#include <kfusion/cuda/projective_icp.hpp>
#include <vector>
#include <string>

#include "g2o/core/sparse_optimizer.h"
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>

namespace kfusion
{
    namespace cuda
    {
        KF_EXPORTS int getCudaEnabledDeviceCount();
        KF_EXPORTS void setDevice(int device);
        KF_EXPORTS std::string getDeviceName(int device);
        KF_EXPORTS bool checkIfPreFermiGPU(int device);
        KF_EXPORTS void printCudaDeviceInfo(int device);
        KF_EXPORTS void printShortCudaDeviceInfo(int device);
    }

    struct KF_EXPORTS KinFuParams
    {
        static KinFuParams default_params();

        int cols;  //pixels
        int rows;  //pixels

        bool integrate_color;  //Color integration
        bool integrate_semantic;  //Semantic integration
        Intr intr;  //Camera parameters

        Vec3i tsdf_volume_dims; //number of voxels for the TSDF volume
        Vec3i color_volume_dims; //number of voxels for the color volume (typically <= TSDF volume)
        Vec3i semantic_volume_dims; //number of voxels for the color volume (typically <= TSDF volume)
        Vec3f volume_size; //meters
        Affine3f volume_pose; //meters, inital pose

        float bilateral_sigma_depth;    //meters
        float bilateral_sigma_spatial;  //pixels
        int   bilateral_kernel_size;    //pixels

        float icp_truncate_depth_dist;  //meters
        float icp_dist_thres;           //meters
        float icp_angle_thres;          //radians
        std::vector<int> icp_iter_num;  //iterations for level index 0,1,..,3

        float tsdf_min_camera_movement; //meters, integrate only if exceedes
        float tsdf_trunc_dist;          //meters;
        int tsdf_max_weight;            //frames

        int color_max_weight;           //frames

        float raycast_step_factor;   // in voxel sizes
        float gradient_delta_factor; // in voxel sizes

        Vec3f light_pose; //meters

    };

    class KF_EXPORTS KinFu
    {
    public:
        typedef cv::Ptr<KinFu> Ptr;

        KinFu(const KinFuParams& params);

        const KinFuParams& params() const;
        KinFuParams& params();

        const cuda::TsdfVolume& tsdf() const;
        cuda::TsdfVolume& tsdf();

        const cv::Ptr<cuda::ColorVolume> color_volume() const;
        cv::Ptr<cuda::ColorVolume> color_volume();

        const cv::Ptr<cuda::SemanticVolume> semantic_volume() const;
        cv::Ptr<cuda::SemanticVolume> semantic_volume();

        const cuda::ProjectiveICP& icp() const;
        cuda::ProjectiveICP& icp();

        void reset();

        int operator()(const cuda::Depth& depth, const cuda::Image& image = cuda::Image(), const cuda::Image& semantic = cuda::Image(),
        				const std::string timestamp = "000000.0000");

        void renderImage(cuda::Image& image, int flags = 0);
        void renderImage(cuda::Image& image, const Affine3f& pose, int flags = 0);

        Affine3f getCameraPose (int time = -1) const;

        void writeKeyframePosesFromGraph(const std::string file_name);
        void clearVolumes();
        void redrawVolumes(const KinFuParams& p);
        void Affine3fToIsometry3d(const cv::Affine3f &from, Eigen::Isometry3d &to);
        g2o::VertexSE3* addVertex(int vertex_id, Eigen::Isometry3d& current_pose);
        void addEdge(int edge_id, g2o::VertexSE3* first_vertex, g2o::VertexSE3* second_vertex, Eigen::Isometry3d& constraint);
        void addEdge(int edge_id, g2o::OptimizableGraph::Vertex* first_vertex, g2o::VertexSE3* second_vertex, Eigen::Isometry3d& constraint);
        void addEdge(int edge_id, g2o::VertexSE3* first_vertex, g2o::OptimizableGraph::Vertex* second_vertex, Eigen::Isometry3d& constraint);
        bool estimateTransform(const cuda::Depth& source_depth, const cuda::Depth& target_depth, cv::Affine3f& transform, const int LEVELS, const KinFuParams& p);

        int getFrameCounter() {return frame_counter_;}

        void storeSubvolume();
        void storePoseVector();
        void savePoseGraph(std::string output_filename);
        void savePoseVector(std::string output_filename);

        void saveEstimatedTrajectories();

        Affine3f getLastSucessPose();

    private:
        void allocate_buffers();

        int frame_counter_;
        KinFuParams params_;

        std::vector<Affine3f> poses_;
        std::vector<std::vector<Affine3f>> vec_poses;

        cuda::Dists dists_;
        cuda::Frame curr_, prev_;

        cuda::Cloud points_;
        cuda::Normals normals_;
        cuda::Depth depths_;
        cuda::Image colors_;
        cuda::Image semantics_;

        cv::Ptr<cuda::TsdfVolume> tsdf_volume_;
        cv::Ptr<cuda::ColorVolume> color_volume_;
        cv::Ptr<cuda::SemanticVolume> semantic_volume_;
        cv::Ptr<cuda::ProjectiveICP> icp_;

        // g2o 
        g2o::SparseOptimizer pose_graph;
        g2o::VertexSE3 *previous_kf_vertex, *current_kf_vertex;
        g2o::VertexSE3 *previous_vertex;
        int vertex_id = 0;
        int edge_id = 0;

        std::vector<cuda::Depth> vec_depth;
        std::vector<cuda::Dists> vec_dist;
        std::vector<cuda::Image> vec_image;
        std::vector<cuda::Image> vec_semantic;
        std::vector<std::string> vec_timestamp;

        std::vector<cuda::Depth> sliding_vec_depth;
        std::vector<int> vec_keyframe_id;

        int pre_keyframe_idx = 0;
        int cur_keyframe_idx = 0;

        // subvolumes
        std::vector<cv::Mat> vec_subvolume;
        std::vector<cv::Mat> vec_subcolorvolume;
    };
}
