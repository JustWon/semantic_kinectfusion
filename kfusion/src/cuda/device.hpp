#pragma once

#include "internal.hpp"
#include "temp_utils.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// TsdfVolume

//__kf_device__
//kfusion::device::TsdfVolume::TsdfVolume(elem_type* _data, int3 _dims, float3 _voxel_size, float _trunc_dist, int _max_weight)
//  : data(_data), dims(_dims), voxel_size(_voxel_size), trunc_dist(_trunc_dist), max_weight(_max_weight) {}

//__kf_device__
//kfusion::device::TsdfVolume::TsdfVolume(const TsdfVolume& other)
//  : data(other.data), dims(other.dims), voxel_size(other.voxel_size), trunc_dist(other.trunc_dist), max_weight(other.max_weight) {}

__kf_device__ kfusion::device::TsdfVolume::elem_type* kfusion::device::TsdfVolume::operator()(int x, int y, int z)
{ return data + x + y*dims.x + z*dims.y*dims.x; }

__kf_device__ const kfusion::device::TsdfVolume::elem_type* kfusion::device::TsdfVolume::operator() (int x, int y, int z) const
{ return data + x + y*dims.x + z*dims.y*dims.x; }

__kf_device__ kfusion::device::TsdfVolume::elem_type* kfusion::device::TsdfVolume::beg(int x, int y) const
{ return data + x + dims.x * y; }

__kf_device__ kfusion::device::TsdfVolume::elem_type* kfusion::device::TsdfVolume::zstep(elem_type *const ptr) const
{ return ptr + dims.x * dims.y; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// ColorVolume

//__kf_device__
//kfusion::device::TsdfVolume::TsdfVolume(elem_type* _data, int3 _dims, float3 _voxel_size, float _trunc_dist, int _max_weight)
//  : data(_data), dims(_dims), voxel_size(_voxel_size), trunc_dist(_trunc_dist), max_weight(_max_weight) {}

//__kf_device__
//kfusion::device::TsdfVolume::TsdfVolume(const TsdfVolume& other)
//  : data(other.data), dims(other.dims), voxel_size(other.voxel_size), trunc_dist(other.trunc_dist), max_weight(other.max_weight) {}

__kf_device__ kfusion::device::ColorVolume::elem_type* kfusion::device::ColorVolume::operator()(int x, int y, int z)
{ return data + x + y*dims.x + z*dims.y*dims.x; }

__kf_device__ const kfusion::device::ColorVolume::elem_type* kfusion::device::ColorVolume::operator() (int x, int y, int z) const
{ return data + x + y*dims.x + z*dims.y*dims.x; }

__kf_device__ kfusion::device::ColorVolume::elem_type* kfusion::device::ColorVolume::beg(int x, int y) const
{ return data + x + dims.x * y; }

__kf_device__ kfusion::device::ColorVolume::elem_type* kfusion::device::ColorVolume::zstep(elem_type *const ptr) const
{ return ptr + dims.x * dims.y; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// SemanticVolume

__kf_device__ kfusion::device::SemanticVolume::elem_type* kfusion::device::SemanticVolume::operator()(int x, int y, int z)
{ return data + x + y*dims.x + z*dims.y*dims.x; }

__kf_device__ const kfusion::device::SemanticVolume::elem_type* kfusion::device::SemanticVolume::operator() (int x, int y, int z) const
{ return data + x + y*dims.x + z*dims.y*dims.x; }

__kf_device__ kfusion::device::SemanticVolume::elem_type* kfusion::device::SemanticVolume::beg(int x, int y) const
{ return data + x + dims.x * y; }

__kf_device__ kfusion::device::SemanticVolume::elem_type* kfusion::device::SemanticVolume::zstep(elem_type *const ptr) const
{ return ptr + dims.x * dims.y; }

__kf_device__ unsigned char* kfusion::device::SemanticVolume::hist_beg(int x, int y) const
{ 
    int class_size = 10;
    return label_histogram + class_size*(x + dims.x * y); 
}

__kf_device__ unsigned char* kfusion::device::SemanticVolume::hist_zstep(unsigned char *const ptr) const
{ 
    int class_size = 10;
    return ptr + class_size*(dims.x * dims.y); 
}

__kf_device__ unsigned char* kfusion::device::SemanticVolume::hist2_beg(int x, int y) const
{ 
    int class_size = 11;
    return label_histogram2 + class_size*(x + dims.x * y); 
}

__kf_device__ unsigned char* kfusion::device::SemanticVolume::hist2_zstep(unsigned char *const ptr) const
{ 
    int class_size = 11;
    return ptr + class_size*(dims.x * dims.y); 
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Projector

__kf_device__ float2 kfusion::device::Projector::operator()(const float3& p) const
{
    float2 coo;
    coo.x = __fmaf_rn(f.x, __fdividef(p.x, p.z), c.x);
    coo.y = __fmaf_rn(f.y, __fdividef(p.y, p.z), c.y);
    return coo;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Reprojector

__kf_device__ float3 kfusion::device::Reprojector::operator()(int u, int v, float z) const
{
    float x = z * (u - c.x) * finv.x;
    float y = z * (v - c.y) * finv.y;
    return make_float3(x, y, z);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// packing/unpacking tsdf volume element

__kf_device__ ushort2 kfusion::device::pack_tsdf (float tsdf, int weight)
{ return make_ushort2 (__float2half_rn (tsdf), weight); }

__kf_device__ float kfusion::device::unpack_tsdf(ushort2 value, int& weight)
{
    weight = value.y;
    return __half2float (value.x);
}
__kf_device__ float kfusion::device::unpack_tsdf (ushort2 value) { return __half2float (value.x); }


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Utility

namespace kfusion
{
    namespace device
    {
        __kf_device__ Vec3f operator*(const Mat3f& m, const Vec3f& v)
        { return make_float3(dot(m.data[0], v), dot (m.data[1], v), dot (m.data[2], v)); }

        __kf_device__ Vec3f operator*(const Aff3f& a, const Vec3f& v) { return a.R * v + a.t; }

        __kf_device__ Vec3f tr(const float4& v) { return make_float3(v.x, v.y, v.z); }

        struct plus
        {
            __kf_device__ float operator () (float l, float r) const  { return l + r; }
            __kf_device__ double operator () (double l, double r) const  { return l + r; }
        };

        struct gmem
        {
            template<typename T> __kf_device__ static T LdCs(T *ptr);
            template<typename T> __kf_device__ static void StCs(const T& val, T *ptr);
        };

        template<> __kf_device__ ushort2 gmem::LdCs(ushort2* ptr);
        template<> __kf_device__ void gmem::StCs(const ushort2& val, ushort2* ptr);
        /*
        template<> __kf_device__ uchar4 gmem::LdCs(uchar4* ptr);
        template<> __kf_device__ void gmem::StCs(const uchar4& val, uchar4* ptr);
         */
    }
}


#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 200

    #if defined(_WIN64) || defined(__LP64__)
        #define _ASM_PTR_ "l"
    #else
        #define _ASM_PTR_ "r"
    #endif

    template<> __kf_device__ ushort2 kfusion::device::gmem::LdCs(ushort2* ptr)
    {
        ushort2 val;
        asm("ld.global.cs.v2.u16 {%0, %1}, [%2];" : "=h"(reinterpret_cast<ushort&>(val.x)), "=h"(reinterpret_cast<ushort&>(val.y)) : _ASM_PTR_(ptr));
        return val;
    }
/*
    template<> __kf_device__ uchar4 kfusion::device::gmem::LdCs(uchar4* ptr)
    {
        uchar4 val;
        asm("ld.global.cs.v2.u8 {%0, %1, %2, %3}, [%4];" : "=h"(reinterpret_cast<uchar&>(val.x)), "=h"(reinterpret_cast<uchar&>(val.y)), "=h"(reinterpret_cast<uchar&>(val.z)), "=h"(reinterpret_cast<uchar&>(val.w)) : _ASM_PTR_(ptr));
        return val;
    }
*/
    template<> __kf_device__ void kfusion::device::gmem::StCs(const ushort2& val, ushort2* ptr)
    {
        short cx = val.x, cy = val.y;
        asm("st.global.cs.v2.u16 [%0], {%1, %2};" : : _ASM_PTR_(ptr), "h"(reinterpret_cast<ushort&>(cx)), "h"(reinterpret_cast<ushort&>(cy)));
    }
/*
    template<> __kf_device__ void kfusion::device::gmem::StCs(const uchar4& val, uchar4* ptr)
    {
        uchar cx = val.x, cy = val.y, cz = val.z, cw = val.w;
        asm("st.global.cs.v2.u8 [%0], {%1, %2, %3, %4};" : : _ASM_PTR_(ptr), cx, cy, cz, cw);
    }
    */
    #undef _ASM_PTR_

#else
    template<> __kf_device__ ushort2 kfusion::device::gmem::LdCs(ushort2* ptr) { return *ptr; }
    template<> __kf_device__ void kfusion::device::gmem::StCs(const ushort2& val, ushort2* ptr) { *ptr = val; }
/*
    template<> __kf_device__ uchar4 kfusion::device::gmem::LdCs(uchar4* ptr) { return *ptr; }
    template<> __kf_device__ void kfusion::device::gmem::StCs(const uchar4& val, uchar4* ptr) { *ptr = val; }
    */
#endif






