//
// Created by Dong-Won Shin on 20180213
//

#include "precomp.hpp"

////////////////////////////////////////////////////////////////////////////////
// TsdfVolume

#pragma mark -
#pragma mark Initialization

/*
 * @name SemanticVolume
 * @fn SemanticVolume (const cv::Vec3i &dims)
 * @brief Constructor of the class
 * @param[in] Number of voxels for each dimensions
 */
kfusion::cuda::SemanticVolume::SemanticVolume(const Vec3i& dims)
    : data_(), trunc_dist_(0.03f), max_weight_(128), dims_(dims),
      size_(Vec3f::all(3.f)), pose_(Affine3f::Identity())
{
    create(dims_);
}

/*
 * @name ~SemanticVolume
 * @fn virtual ~SemanticVolume(void)
 * @brief Destructor of the class
 */
kfusion::cuda::SemanticVolume::~SemanticVolume() {}

/*
 * @name create
 * @fn void create(const Vec3i& dims)
 * @brief Initialize the volume on the device
 * @param[in] Number of voxels for each dimensions
 */
void kfusion::cuda::SemanticVolume::create(const Vec3i& dims)
{
    dims_ = dims;
    int voxels_number = dims_[0] * dims_[1] * dims_[2];
    data_.create(voxels_number * 4 * sizeof(unsigned char));
    setTruncDist(trunc_dist_);

    int class_size1=10;
    int class_size2=11;
    label_histogram_.create(voxels_number*class_size1*sizeof(unsigned char));
    label_histogram2_.create(voxels_number*class_size2*sizeof(unsigned char));
    clear();
}

#pragma mark -
#pragma mark Getters and Setters

void kfusion::cuda::SemanticVolume::setTruncDist(float distance)
{
    Vec3f vsz = getVoxelSize();
    float max_coeff = std::max<float>(std::max<float>(vsz[0], vsz[1]), vsz[2]);
    trunc_dist_ = std::max (distance, 2.1f * max_coeff);
}

#pragma mark -
#pragma mark Usage

/*
 * @name clear
 * @fn virtual void clear()
 * @brief Allocate memory on device and initialize at 0
 */
void kfusion::cuda::SemanticVolume::clear()
{
    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());

    device::SemanticVolume volume(data_.ptr<uchar4>(), label_histogram_.ptr<uchar>(), label_histogram2_.ptr<uchar>(), dims, vsz, trunc_dist_, max_weight_);
    device::clear_volume(volume);
}

/*
 * @name integrate
 * @fn virtual void integrate(const Image& rgb_image, const Affine3f& camera_pose, const Intr& intr)
 * @brief
 * @param[in] rgb_image, the new frame to integrate
 * @param[in] depth_map, the raycasted depth map
 * @param[in] camera_pose, the current pose of the camera
 * @param[in] intr, the intrinsic parameters of the RGB camera
 */
void kfusion::cuda::SemanticVolume::integrate(const Image& rgb_image,
                                           const Dists& depth_map,
                                           const Affine3f& camera_pose,
                                           const Intr& intr)
{
    Affine3f vol2cam = camera_pose.inv() * pose_;

    device::Projector proj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff = device_cast<device::Aff3f>(vol2cam);

    device::SemanticVolume volume(data_.ptr<uchar4>(), label_histogram_.ptr<uchar>(), label_histogram2_.ptr<uchar>(), dims, vsz, trunc_dist_, max_weight_);
    device::integrate(rgb_image, depth_map, volume, aff, proj);
}

/*
 * @name fetchSemantics
 * @fn void fetchSemantics(const DeviceArray<Point>& cloud, DeviceArray<RGB>& semantics) const
 * @brief Gets semantic for a point cloud
 * @param[in] cloud, the coordinates of the semantics to extract
 * @param[in] semantics, the semantics stored in the volume
 */
void kfusion::cuda::SemanticVolume::fetchSemantics(const DeviceArray<Point>& cloud,
                                             DeviceArray<RGB>& semantics) const
{
    if (semantics.size() != cloud.size())
        semantics.create (cloud.size());

    DeviceArray<device::Point>& pts = (DeviceArray<device::Point>&)cloud;
    PtrSz<uchar4> col(reinterpret_cast<uchar4*>(semantics.ptr()), semantics.size());
    // DeviceArray<device::Color>& col = (DeviceArray<device::Color>&)semantics;

    Affine3f pose_inv = pose_.inv();

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff_inv  = device_cast<device::Aff3f>(pose_inv);

    device::SemanticVolume volume((uchar4*)data_.ptr<uchar4>(), (uchar*)label_histogram_.ptr<uchar>(), (uchar*)label_histogram2_.ptr<uchar>(), dims, vsz, trunc_dist_, max_weight_);
    device::fetchSemantics(volume, aff_inv, pts, col);
}
