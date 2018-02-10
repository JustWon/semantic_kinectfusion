#include "device.hpp"
#include "texture_binder.hpp"
#include "../internal.hpp"
#include <stdio.h>
#include <cmath>

using namespace kfusion::device;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Volume initialization

namespace kfusion
{
    namespace device
    {
        __global__ void clear_volume_kernel(ColorVolume color)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x < color.dims.x && y < color.dims.y)
            {
                uchar4 *beg = color.beg(x, y);
                uchar4 *end = beg + color.dims.x * color.dims.y * color.dims.z;

                for(uchar4* pos = beg; pos != end; pos = color.zstep(pos))
                    *pos = make_uchar4 (0, 0, 0, 0);
            }
        }
    }
}

void kfusion::device::clear_volume(ColorVolume volume)
{
    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = divUp (volume.dims.x, block.x);
    grid.y = divUp (volume.dims.y, block.y);

    clear_volume_kernel<<<grid, block>>>(volume);
    cudaSafeCall ( cudaGetLastError () );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Volume integration

namespace kfusion
{
    namespace device
    {
        texture<uchar4, 2> image_tex(0, cudaFilterModePoint, cudaAddressModeBorder);
        texture<float, 2> depth_tex(0, cudaFilterModePoint, cudaAddressModeBorder, cudaCreateChannelDescHalf());

        struct ColorIntegrator {
            Aff3f vol2cam;
            PtrStep<float> vmap;
            Projector proj;
            int2 im_size;

            float tranc_dist_inv;

            __kf_device__
            void operator()(ColorVolume& volume) const
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= volume.dims.x || y >= volume.dims.y)
                    return;

                float3 zstep = make_float3(vol2cam.R.data[0].z, vol2cam.R.data[1].z, vol2cam.R.data[2].z) * volume.voxel_size.z;

                float3 vx = make_float3(x * volume.voxel_size.x, y * volume.voxel_size.y, 0);
                float3 vc = vol2cam * vx; //tranform from volume coo frame to camera one

                ColorVolume::elem_type* vptr = volume.beg(x, y);
                for(int i = 0; i < volume.dims.z; ++i, vc += zstep, vptr = volume.zstep(vptr))
                {
                    float2 coo = proj(vc); // project to image coordinate
                    // check wether coo in inside the image boundaries
                    if (coo.x >= 0.0 && coo.y >= 0.0 &&
                        coo.x < im_size.x && coo.y < im_size.y) {

                        float Dp = tex2D(depth_tex, coo.x, coo.y);
                        if(Dp == 0 || vc.z <= 0)
                            continue;

                        bool update = false;
                        // Check the distance
                        float sdf = Dp - sqrt(dot(vc, vc)); //Dp - norm(v)
                        update = (sdf > -volume.trunc_dist) && (sdf < volume.trunc_dist);
                        if (update)
                        {
                            // Read the existing value and weight
                            uchar4 volume_rgbw = *vptr;
                            int weight_prev = volume_rgbw.w;

                            // Average with new value and weight
                            uchar4 rgb = tex2D(image_tex, coo.x, coo.y);

                            const float Wrk = 1.f;
                            float new_x =  __fdividef(__fmaf_rn(volume_rgbw.x, weight_prev, rgb.x), weight_prev + Wrk);
                            //uchar new_x = (volume_rgbw.x * weight_prev + Wrk * rgb.x) / (weight_prev + Wrk);
                            float new_y =  __fdividef(__fmaf_rn(volume_rgbw.y, weight_prev, rgb.y), weight_prev + Wrk);
                            //uchar new_y = (volume_rgbw.y * weight_prev + Wrk * rgb.y) / (weight_prev + Wrk);
                            float new_z =  __fdividef(__fmaf_rn(volume_rgbw.z, weight_prev, rgb.z), weight_prev + Wrk);
                            //uchar new_z = (volume_rgbw.z * weight_prev + Wrk * rgb.z) / (weight_prev + Wrk);

                            int weight_new = min(weight_prev + 1, 255);

                            uchar4 volume_rgbw_new;
                            volume_rgbw_new.x = (uchar)__float2int_rn(new_x);
                            volume_rgbw_new.y = (uchar)__float2int_rn(new_y);
                            volume_rgbw_new.z = (uchar)__float2int_rn(new_z);
                            volume_rgbw_new.w = min(volume.max_weight, weight_new);

                            // Write back
                            *vptr = volume_rgbw_new;
                        }
                    } // in camera image range
                } // for (int i=0; i<volume.dims.z; ++i, vc += zstep, vptr = volume.zstep(vptr))
            } // void operator()
        };

        __global__ void integrate_kernel(const ColorIntegrator integrator, ColorVolume volume) {integrator(volume);};
    }
}

void kfusion::device::integrate(const PtrStepSz<uchar4>& rgb_image,
                                const PtrStepSz<ushort>& depth_map,
                                ColorVolume& volume,
                                const Aff3f& aff,
                                const Projector& proj)
{
    ColorIntegrator ti;
    ti.im_size = make_int2(rgb_image.cols, rgb_image.rows);
    ti.vol2cam = aff;
    ti.proj = proj;
    ti.tranc_dist_inv = 1.f/volume.trunc_dist;

    image_tex.filterMode = cudaFilterModePoint;
    image_tex.addressMode[0] = cudaAddressModeBorder;
    image_tex.addressMode[1] = cudaAddressModeBorder;
    image_tex.addressMode[2] = cudaAddressModeBorder;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    TextureBinder image_binder(rgb_image, image_tex, channelDesc);// (void)image_binder;

    depth_tex.filterMode = cudaFilterModePoint;
    depth_tex.addressMode[0] = cudaAddressModeBorder;
    depth_tex.addressMode[1] = cudaAddressModeBorder;
    depth_tex.addressMode[2] = cudaAddressModeBorder;
    TextureBinder depth_binder(depth_map, depth_tex, cudaCreateChannelDescHalf());// (void)depth_binder;

    dim3 block(32, 8);
    dim3 grid(divUp(volume.dims.x, block.x), divUp(volume.dims.y, block.y));

    integrate_kernel<<<grid, block>>>(ti, volume);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall ( cudaDeviceSynchronize() );
}

namespace kfusion
{
    namespace device
    {
        struct ColorFetcher
        {
            ColorVolume volume;

            float3 cell_size;
            int n_pts;
            const float4* pts_data;
            Aff3f aff_inv;

            ColorFetcher(const ColorVolume& volume, float3 cell_size);

            __kf_device__
            void operator()(PtrSz<Color> colors) const
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx < n_pts)
                {
                  int3 v;
                  float4 p = *(const float4 *) (pts_data + idx);
                  float3 px = make_float3(p.x, p.y, p.z);
                  float3 pv = aff_inv * px;
                  v.x = __float2int_rd(pv.x / cell_size.x);        // round to negative infinity
                  v.y = __float2int_rd(pv.y / cell_size.y);
                  v.z = __float2int_rd(pv.z / cell_size.z);

                  uchar4 rgbw = *volume(v.x, v.y, v.z);
                  uchar4 *pix = colors.data;
                  pix[idx] = rgbw; //bgra
                }
            }
        };

        inline ColorFetcher::ColorFetcher(const ColorVolume& _volume, float3 _cell_size)
        : volume(_volume), cell_size(_cell_size) {}

        __global__ void fetchColors_kernel (const ColorFetcher colorfetcher, PtrSz<Color> colors)
        {colorfetcher(colors);};
    }
}

void
kfusion::device::fetchColors(const ColorVolume& volume, const Aff3f& aff_inv, const PtrSz<Point>& points, PtrSz<Color>& colors)
{
    const int block = 256;

    if (points.size != colors.size || points.size == 0)
        return;

    float3 cell_size = make_float3 (volume.voxel_size.x, volume.voxel_size.y, volume.voxel_size.z);

    ColorFetcher cf(volume, cell_size);
    cf.n_pts = points.size;
    cf.pts_data = points.data;
    cf.aff_inv = aff_inv;

    fetchColors_kernel<<<divUp (points.size, block), block>>>(cf, colors);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
};

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// /// Volume ray casting

// namespace kfusion
// {
//     namespace device
//     {
//         __kf_device__ void intersect(float3 ray_org, float3 ray_dir, /*float3 box_min,*/ float3 box_max, float &tnear, float &tfar)
//         {
//             const float3 box_min = make_float3(0.f, 0.f, 0.f);

//             // compute intersection of ray with all six bbox planes
//             float3 invR = make_float3(1.f/ray_dir.x, 1.f/ray_dir.y, 1.f/ray_dir.z);
//             float3 tbot = invR * (box_min - ray_org);
//             float3 ttop = invR * (box_max - ray_org);

//             // re-order intersections to find smallest and largest on each axis
//             float3 tmin = make_float3(fminf(ttop.x, tbot.x), fminf(ttop.y, tbot.y), fminf(ttop.z, tbot.z));
//             float3 tmax = make_float3(fmaxf(ttop.x, tbot.x), fmaxf(ttop.y, tbot.y), fmaxf(ttop.z, tbot.z));

//             // find the largest tmin and the smallest tmax
//             tnear = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
//             tfar  = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));
//         }

//         template<typename Vol>
//         __kf_device__ float interpolate(const Vol& volume, const float3& p_voxels)
//         {
//             float3 cf = p_voxels;

//             //rounding to negative infinity
//             int3 g = make_int3(__float2int_rd (cf.x), __float2int_rd (cf.y), __float2int_rd (cf.z));

//             if (g.x < 0 || g.x >= volume.dims.x - 1 || g.y < 0 || g.y >= volume.dims.y - 1 || g.z < 0 || g.z >= volume.dims.z - 1)
//                 return numeric_limits<float>::quiet_NaN();

//             float a = cf.x - g.x;
//             float b = cf.y - g.y;
//             float c = cf.z - g.z;

//             float tsdf = 0.f;
//             tsdf += unpack_tsdf(*volume(g.x + 0, g.y + 0, g.z + 0)) * (1 - a) * (1 - b) * (1 - c);
//             tsdf += unpack_tsdf(*volume(g.x + 0, g.y + 0, g.z + 1)) * (1 - a) * (1 - b) *      c;
//             tsdf += unpack_tsdf(*volume(g.x + 0, g.y + 1, g.z + 0)) * (1 - a) *      b  * (1 - c);
//             tsdf += unpack_tsdf(*volume(g.x + 0, g.y + 1, g.z + 1)) * (1 - a) *      b  *      c;
//             tsdf += unpack_tsdf(*volume(g.x + 1, g.y + 0, g.z + 0)) *      a  * (1 - b) * (1 - c);
//             tsdf += unpack_tsdf(*volume(g.x + 1, g.y + 0, g.z + 1)) *      a  * (1 - b) *      c;
//             tsdf += unpack_tsdf(*volume(g.x + 1, g.y + 1, g.z + 0)) *      a  *      b  * (1 - c);
//             tsdf += unpack_tsdf(*volume(g.x + 1, g.y + 1, g.z + 1)) *      a  *      b  *      c;
//             return tsdf;
//         }

//         struct ColorRaycaster
//         {
//             ColorVolume color_volume;
//             TsdfVolume tsdf_volume;

//             Aff3f aff;
//             Mat3f Rinv;

//             Vec3f volume_size;
//             Reprojector reproj;
//             float time_step;
//             float3 gradient_delta;
//             float3 voxel_size_inv;

//             ColorRaycaster(const ColorVolume& color_volume, const TsdfVolume& volume, const Aff3f& aff, const Mat3f& Rinv, const Reprojector& _reproj);

//             __kf_device__
//             float fetch_tsdf(const float3& p) const
//             {
//                 //rounding to nearest even
//                 int x = __float2int_rn (p.x * voxel_size_inv.x);
//                 int y = __float2int_rn (p.y * voxel_size_inv.y);
//                 int z = __float2int_rn (p.z * voxel_size_inv.z);
//                 return unpack_tsdf(*tsdf_volume(x, y, z));
//             }

//             __kf_device__
//             void operator()(PtrStepSz<Point> colors) const
//             {
//                 int x = blockIdx.x * blockDim.x + threadIdx.x;
//                 int y = blockIdx.y * blockDim.y + threadIdx.y;

//                 if (x >= colors.cols || y >= colors.rows)
//                     return;

//                 const float qnan = numeric_limits<float>::quiet_NaN();

//                 colors(y, x) = make_float4(qnan, qnan, qnan, qnan);

//                 float3 ray_org = aff.t;
//                 float3 ray_dir = normalized( aff.R * reproj(x, y, 1.f) );

//                 // We do subtract voxel size to minimize checks after
//                 // Note: origin of volume coordinate is placeed
//                 // in the center of voxel (0,0,0), not in the corener of the voxel!
//                 float3 box_max = volume_size - tsdf_volume.voxel_size;

//                 float tmin, tmax;
//                 intersect(ray_org, ray_dir, box_max, tmin, tmax);

//                 const float min_dist = 0.f;
//                 tmin = fmax(min_dist, tmin);
//                 if (tmin >= tmax)
//                     return;

//                 tmax -= time_step;
//                 float3 vstep = ray_dir * time_step;
//                 float3 next = ray_org + ray_dir * tmin;

//                 float tsdf_next = fetch_tsdf(next);
//                 for (float tcurr = tmin; tcurr < tmax; tcurr += time_step)
//                 {
//                     float tsdf_curr = tsdf_next;
//                     float3     curr = next;
//                     next += vstep;

//                     tsdf_next = fetch_tsdf(next);
//                     if (tsdf_curr < 0.f && tsdf_next > 0.f)
//                         break;

//                     if (tsdf_curr > 0.f && tsdf_next < 0.f)
//                     {
//                         float Ft   = interpolate(tsdf_volume, curr * voxel_size_inv);
//                         float Ftdt = interpolate(tsdf_volume, next * voxel_size_inv);

//                         float Ts = tcurr - __fdividef(time_step * Ft, Ftdt - Ft);

//                         float3 vertex = ray_org + ray_dir * Ts;

//                         colors(y, x) = make_float4(1,1,1,0);
//                         break;
//                     }
//                 } /* for (;;) */
//             }
//         };

//         inline ColorRaycaster::ColorRaycaster(const ColorVolume& color_volume, const TsdfVolume& tsdf_volume, const Aff3f& _aff, const Mat3f& _Rinv, const Reprojector& _reproj)
//             : color_volume(color_volume), tsdf_volume(tsdf_volume), aff(_aff), Rinv(_Rinv), reproj(_reproj) {}

//         __global__ void raycast_kernel(const ColorRaycaster raycaster, PtrStepSz<Point> colors)
//         { raycaster(colors); };

//     }
// }

// void kfusion::device::raycast(const ColorVolume& color_volume, const TsdfVolume& tsdf_volume, const Aff3f& aff, const Mat3f& Rinv, const Reprojector& reproj,
//                               Points& colors, float raycaster_step_factor, float gradient_delta_factor)
// {
//     ColorRaycaster rc(color_volume, tsdf_volume, aff, Rinv, reproj);

//     rc.volume_size = color_volume.voxel_size * color_volume.dims;
//     rc.time_step = color_volume.trunc_dist * raycaster_step_factor;
//     rc.gradient_delta = color_volume.voxel_size * gradient_delta_factor;
//     rc.voxel_size_inv = 1.f/color_volume.voxel_size;

//     dim3 block(32, 8);
//     dim3 grid (divUp (colors.cols(), block.x), divUp (colors.rows(), block.y));

//     // raycast_kernel<<<grid, block>>>(rc, (PtrStepSz<Point>)colors);
//     // cudaSafeCall (cudaGetLastError ());
// }
