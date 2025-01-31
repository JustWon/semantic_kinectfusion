#pragma once

#include <kfusion/cuda/device_array.hpp>
#include "safe_call.hpp"
#include "kfusion/types.hpp"

namespace kfusion
{
    namespace device
    {
        typedef float4 Normal;
        typedef float4 Point;

        typedef unsigned short ushort;
        typedef unsigned char uchar;

        typedef uchar4 Color;

        typedef PtrStepSz<ushort> Dists;
        typedef DeviceArray2D<ushort> Depth;
        typedef DeviceArray2D<Normal> Normals;
        typedef DeviceArray2D<Point> Points;
        typedef PtrStepSz<Color> Colors;  // Not sure it's needed
        typedef DeviceArray2D<Color> Image;

        typedef int3   Vec3i;
        typedef float3 Vec3f;
        struct Mat3f { float3 data[3]; };
        struct Aff3f { Mat3f R; Vec3f t; };

        struct TsdfVolume
        {
        public:
            typedef ushort2 elem_type;

            elem_type *const data;
            const int3 dims;
            const float3 voxel_size;
            const float trunc_dist;
            const int max_weight;

            TsdfVolume(elem_type* data, int3 dims, float3 voxel_size, float trunc_dist, int max_weight);
            //TsdfVolume(const TsdfVolume&);

            __kf_device__ elem_type* operator()(int x, int y, int z);
            __kf_device__ const elem_type* operator() (int x, int y, int z) const ;
            __kf_device__ elem_type* beg(int x, int y) const;
            __kf_device__ elem_type* zstep(elem_type *const ptr) const;
        private:
            TsdfVolume& operator=(const TsdfVolume&);
        };

        struct ColorVolume
        {
        public:
            typedef uchar4 elem_type;

            elem_type *const data;
            const int3 dims;
            const float3 voxel_size;
            const float trunc_dist;
            const int max_weight;

            ColorVolume(elem_type* data, int3 dims, float3 voxel_size, float trunc_dist, int max_weight);
            //TsdfVolume(const TsdfVolume&);

            __kf_device__ elem_type* operator()(int x, int y, int z);
            __kf_device__ const elem_type* operator() (int x, int y, int z) const ;
            __kf_device__ elem_type* beg(int x, int y) const;
            __kf_device__ elem_type* zstep(elem_type *const ptr) const;
        private:
            ColorVolume& operator=(const ColorVolume&);
        };

        struct SemanticVolume
        {
        public:
            typedef uchar4 elem_type;

            elem_type *const data;
            uchar *const label_histogram;
            uchar *const label_histogram2;
            const int3 dims;
            const float3 voxel_size;
            const float trunc_dist;
            const int max_weight;

            SemanticVolume(elem_type* data, uchar* label_histogram, uchar* label_histogram2, int3 dims, float3 voxel_size, float trunc_dist, int max_weight);
            // change precomp.hpp too

            __kf_device__ elem_type* operator()(int x, int y, int z);
            __kf_device__ const elem_type* operator() (int x, int y, int z) const ;
            __kf_device__ elem_type* beg(int x, int y) const;
            __kf_device__ elem_type* zstep(elem_type *const ptr) const;
            __kf_device__ unsigned char* hist_beg(int x, int y) const;
            __kf_device__ unsigned char* hist_zstep(unsigned char *const ptr) const;
            __kf_device__ unsigned char* hist2_beg(int x, int y) const;
            __kf_device__ unsigned char* hist2_zstep(unsigned char *const ptr) const;
        private:
            SemanticVolume& operator=(const SemanticVolume&);
        };

        struct Projector
        {
            float2 f, c;
            Projector(){}
            Projector(float fx, float fy, float cx, float cy);
            __kf_device__ float2 operator()(const float3& p) const;
        };

        struct Reprojector
        {
            Reprojector() {}
            Reprojector(float fx, float fy, float cx, float cy);
            float2 finv, c;
            __kf_device__ float3 operator()(int x, int y, float z) const;
        };

        struct ComputeIcpHelper
        {
            struct Policy;
            struct PageLockHelper
            {
                float* data;
                PageLockHelper();
                ~PageLockHelper();
            };

            float min_cosine;
            float dist2_thres;

            Aff3f aff;

            float rows, cols;
            float2 f, c, finv;

            PtrStep<ushort> dcurr;
            PtrStep<Normal> ncurr;
            PtrStep<Point> vcurr;

            ComputeIcpHelper(float dist_thres, float angle_thres);
            void setLevelIntr(int level_index, float fx, float fy, float cx, float cy);

            void operator()(const Depth& dprev, const Normals& nprev, DeviceArray2D<float>& buffer, float* data, cudaStream_t stream);
            void operator()(const Points& vprev, const Normals& nprev, DeviceArray2D<float>& buffer, float* data, cudaStream_t stream);

            static void allocate_buffer(DeviceArray2D<float>& buffer, int partials_count = -1);

            //private:
            __kf_device__ int find_coresp(int x, int y, float3& n, float3& d, float3& s) const;
            __kf_device__ void partial_reduce(const float row[7], PtrStep<float>& partial_buffer) const;
            __kf_device__ float2 proj(const float3& p) const;
            __kf_device__ float3 reproj(float x, float y, float z)  const;
        };

        //tsdf volume functions
        void clear_volume(TsdfVolume volume);
        void integrate(const Dists& depth, TsdfVolume& volume, const Aff3f& aff, const Projector& proj);

        void raycast(const TsdfVolume& volume, const Aff3f& aff, const Mat3f& Rinv,
                     const Reprojector& reproj, Depth& depth, Normals& normals, float step_factor, float delta_factor);

        void raycast(const TsdfVolume& volume, const Aff3f& aff, const Mat3f& Rinv,
                     const Reprojector& reproj, Points& points, Normals& normals, Image& colors, Image& semantics, float step_factor, float delta_factor, 
                     const uchar4* color_vol_data, const uchar4* semantic_vol_data);

        __kf_device__ ushort2 pack_tsdf(float tsdf, int weight);
        __kf_device__ float unpack_tsdf(ushort2 value, int& weight);
        __kf_device__ float unpack_tsdf(ushort2 value);

        //color volume functions
        void clear_volume(ColorVolume volume);
        void integrate(const Colors& image, const Dists& depth_map, ColorVolume& volume, const Aff3f& aff, const Projector& proj);
        void fetchColors(const ColorVolume& volume, const Aff3f& aff_inv, const PtrSz<Point>& points, PtrSz<Color>& colors);

        //semantic volume functions
        void clear_volume(SemanticVolume volume);
        void integrate(const Colors& image, const Dists& depth_map, SemanticVolume& volume, const Aff3f& aff, const Projector& proj);
        void fetchSemantics(const SemanticVolume& volume, const Aff3f& aff_inv, const PtrSz<Point>& points, PtrSz<Color>& semantics);

        //image proc functions
        void compute_dists(const Depth& depth, Dists dists, float2 f, float2 c);

        void truncateDepth(Depth& depth, float max_dist /*meters*/);
        void bilateralFilter(const Depth& src, Depth& dst, int kernel_size, float sigma_spatial, float sigma_depth);
        void depthPyr(const Depth& source, Depth& pyramid, float sigma_depth);

        void resizeDepthNormals(const Depth& depth, const Normals& normals, Depth& depth_out, Normals& normals_out);
        void resizePointsNormals(const Points& points, const Normals& normals, Points& points_out, Normals& normals_out);

        void computeNormalsAndMaskDepth(const Reprojector& reproj, Depth& depth, Normals& normals);
        void computePointNormals(const Reprojector& reproj, const Depth& depth, Points& points, Normals& normals);

        void renderImage(const Depth& depth, const Normals& normals, const Reprojector& reproj, const Vec3f& light_pose, Image& image);
        void renderImage(const Points& points, const Normals& normals, const Reprojector& reproj, const Vec3f& light_pose, Image& image);
        void renderTangentColors(const Normals& normals, Image& image);

        //exctraction functionality
        size_t extractCloud(const TsdfVolume& volume, const Aff3f& aff, PtrSz<Point> output);
        void extractNormals(const TsdfVolume& volume, const PtrSz<Point>& points, const Aff3f& aff, const Mat3f& Rinv, float gradient_delta_factor, float4* output);

        struct float8  { float x, y, z, w, c1, c2, c3, c4; };
        struct float12 { float x, y, z, w, normal_x, normal_y, normal_z, n4, c1, c2, c3, c4; };
        void mergePointNormal(const DeviceArray<Point>& cloud, const DeviceArray<float8>& normals, const DeviceArray<float12>& output);

        //marching cubes functions
        void bindTextures(const int *edgeBuf, const int *triBuf, const int *numVertsBuf);
        void unbindTextures();
        int getOccupiedVoxels(const TsdfVolume& volume, DeviceArray2D<int>& occupied_voxels);
        int computeOffsetsAndTotalVertices(DeviceArray2D<int>& occupied_voxels);
        void generateTriangles(const TsdfVolume& volume, const Aff3f& aff, const DeviceArray2D<int>& occupied_voxels, DeviceArray<Point>& output);
    }
}
