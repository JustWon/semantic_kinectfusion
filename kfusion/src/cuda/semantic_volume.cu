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
        __global__ void clear_volume_kernel(SemanticVolume semantic)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x < semantic.dims.x && y < semantic.dims.y)
            {
                uchar4 *beg = semantic.beg(x, y);
                uchar4 *end = beg + semantic.dims.x * semantic.dims.y * semantic.dims.z;

                for(uchar4* pos = beg; pos != end; pos = semantic.zstep(pos))
                    *pos = make_uchar4 (0, 0, 0, 0);

                ////////////////////
                int class_size = 10;
                uchar *hist_beg = semantic.hist_beg(x, y);
                uchar *hist_end = hist_beg + class_size*(semantic.dims.x * semantic.dims.y * semantic.dims.z);

                for(uchar* hist_pos = hist_beg; hist_pos != hist_end; hist_pos = semantic.hist_zstep(hist_pos))
                    for(int i = 0 ; i < class_size ; i++)
                        *(hist_beg+i) = 0;
            }
        }
    }
}

void kfusion::device::clear_volume(SemanticVolume volume)
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

        struct SenamticIntegrator {
            Aff3f vol2cam;
            PtrStep<float> vmap;
            Projector proj;
            int2 im_size;

            float tranc_dist_inv;

            int palette[22][3] = {
                {0, 0, 0},
                {128, 0, 0},
                {0, 128, 0},
                {128, 128, 0},
                {0, 0, 128},
                {128, 0, 128},
                {0, 128, 128},
                {128, 128, 128},
                {64, 0, 0},
                {192, 0, 0},
                {64, 128, 0},
                {192, 128, 0},
                {64, 0, 128},
                {192, 0, 128},
                {64, 128, 128},
                {192, 128, 128},
                {0, 64, 0},
                {128, 64, 0},
                {0, 192, 0},
                {128, 192, 0},
                {0, 64, 12}
            };

            __kf_device__
            uchar color2label(uchar4 color) const
            {
                for(int i = 0 ; i < 21 ; i++)
                {
                    if (color.x == palette[i][0] &&
                        color.y == palette[i][1] &&
                        color.z == palette[i][2])
                    {
                        return i;
                    }
                }
                return 0;
            }

            __kf_device__
            uchar4 label2color(uchar label) const
            {
                uchar4 color = make_uchar4(palette[label][0], palette[label][1], palette[label][2], 0);
                return color;
            }

            __kf_device__
            void operator()(SemanticVolume& volume) const
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= volume.dims.x || y >= volume.dims.y)
                    return;

                float3 zstep = make_float3(vol2cam.R.data[0].z, vol2cam.R.data[1].z, vol2cam.R.data[2].z) * volume.voxel_size.z;

                float3 vx = make_float3(x * volume.voxel_size.x, y * volume.voxel_size.y, 0);
                float3 vc = vol2cam * vx; //tranform from volume coo frame to camera one

                SemanticVolume::elem_type* vptr = volume.beg(x, y);
                unsigned char* hptr = volume.hist_beg(x, y);
                for(int i = 0; i < volume.dims.z; ++i, vc += zstep, vptr = volume.zstep(vptr), hptr = volume.hist_zstep(hptr))
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

                            uchar class_idx = color2label(rgb);
                            uchar count = *((uchar*)hptr+class_idx);
                            *((uchar*)hptr+class_idx) = count + 1;

                            int class_size = 10;
                            uchar *hist_pointer = (uchar*)hptr;
                            int max_cnt = -1000; uchar max_idx = -1;
                            
                            for (int cur_idx = 0 ; cur_idx < class_size ; cur_idx++)
                            {
                                uchar cur_cnt = *(hist_pointer+cur_idx);
                                if(cur_cnt > max_cnt)
                                {
                                    max_cnt = cur_cnt;
                                    max_idx = cur_idx;
                                }
                            }

                            uchar4 class_color = label2color(max_idx);
                            *vptr = class_color;

                            // float new_x =  __fdividef(__fmaf_rn(volume_rgbw.x, weight_prev, rgb.x), weight_prev + Wrk);
                            // //uchar new_x = (volume_rgbw.x * weight_prev + Wrk * rgb.x) / (weight_prev + Wrk);
                            // float new_y =  __fdividef(__fmaf_rn(volume_rgbw.y, weight_prev, rgb.y), weight_prev + Wrk);
                            // //uchar new_y = (volume_rgbw.y * weight_prev + Wrk * rgb.y) / (weight_prev + Wrk);
                            // float new_z =  __fdividef(__fmaf_rn(volume_rgbw.z, weight_prev, rgb.z), weight_prev + Wrk);
                            // //uchar new_z = (volume_rgbw.z * weight_prev + Wrk * rgb.z) / (weight_prev + Wrk);

                            // int weight_new = min(weight_prev + 1, 255);

                            // uchar4 volume_rgbw_new;
                            // volume_rgbw_new.x = (uchar)__float2int_rn(new_x);
                            // volume_rgbw_new.y = (uchar)__float2int_rn(new_y);
                            // volume_rgbw_new.z = (uchar)__float2int_rn(new_z);
                            // volume_rgbw_new.w = min(volume.max_weight, weight_new);

                            // // Write back
                            // *vptr = volume_rgbw_new;
                        }
                    } // in camera image range
                } // for (int i=0; i<volume.dims.z; ++i, vc += zstep, vptr = volume.zstep(vptr))
            } // void operator()
        };

        __global__ void integrate_kernel(const SenamticIntegrator integrator, SemanticVolume volume) {integrator(volume);};
    }
}

void kfusion::device::integrate(const PtrStepSz<uchar4>& rgb_image,
                                const PtrStepSz<ushort>& depth_map,
                                SemanticVolume& volume,
                                const Aff3f& aff,
                                const Projector& proj)
{
    SenamticIntegrator ti;
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
            SemanticVolume volume;

            float3 cell_size;
            int n_pts;
            const float4* pts_data;
            Aff3f aff_inv;

            ColorFetcher(const SemanticVolume& volume, float3 cell_size);

            __kf_device__
            void operator()(PtrSz<Color> semantics) const
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
                  uchar4 *pix = semantics.data;
                  pix[idx] = rgbw; //bgra
                }
            }
        };

        inline ColorFetcher::ColorFetcher(const SemanticVolume& _volume, float3 _cell_size)
        : volume(_volume), cell_size(_cell_size) {}

        __global__ void fetchSemantics_kernel (const ColorFetcher semanticfetcher, PtrSz<Color> semantics)
        {semanticfetcher(semantics);};
    }
}

void
kfusion::device::fetchSemantics(const SemanticVolume& volume, const Aff3f& aff_inv, const PtrSz<Point>& points, PtrSz<Color>& semantics)
{
    const int block = 256;

    if (points.size != semantics.size || points.size == 0)
        return;

    float3 cell_size = make_float3 (volume.voxel_size.x, volume.voxel_size.y, volume.voxel_size.z);

    ColorFetcher cf(volume, cell_size);
    cf.n_pts = points.size;
    cf.pts_data = points.data;
    cf.aff_inv = aff_inv;

    fetchSemantics_kernel<<<divUp (points.size, block), block>>>(cf, semantics);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
};