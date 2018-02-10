#include "device.hpp"
#include "texture_binder.hpp"
#include "../internal.hpp"

#include "thrust/device_ptr.h"
#include "thrust/scan.h"

namespace kfusion
{
    namespace device
    {
        //texture<int, 1, cudaReadModeElementType> edgeTex;
        texture<int, 1, cudaReadModeElementType> triTex;
        texture<int, 1, cudaReadModeElementType> numVertsTex;
    }
}

void kfusion::device::bindTextures (const int */*edgeBuf*/, const int *triBuf, const int *numVertsBuf)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    //cudaSafeCall(cudaBindTexture(0, edgeTex, edgeBuf, desc) );
    cudaSafeCall (cudaBindTexture (0, triTex, triBuf, desc) );
    cudaSafeCall (cudaBindTexture (0, numVertsTex, numVertsBuf, desc) );
}

void kfusion::device::unbindTextures ()
{
    //cudaSafeCall( cudaUnbindTexture(edgeTex) );
    cudaSafeCall ( cudaUnbindTexture (numVertsTex) );
    cudaSafeCall ( cudaUnbindTexture (triTex) );
}

namespace kfusion
{
    namespace device
    {
        __device__ int global_count = 0;
        __device__ int output_count;
        __device__ unsigned int blocks_done = 0;

        struct CubeIndexEstimator
        {
            TsdfVolume volume;

            CubeIndexEstimator(const TsdfVolume& _volume) : volume(_volume) {}

            static __kf_device__ float isoValue () {return 0.f;}

            __kf_device__
            void readTsdf (int x, int y, int z, float &tsdf, int &weight) const
            {
                tsdf = unpack_tsdf(*volume(x, y, z), weight);
            }

            __kf_device__
            int computeCubeIndex (int x, int y, int z, float f[8]) const
            {
                int weight;
                readTsdf(x, y, z, f[0], weight);
                if (weight == 0) return 0;
                readTsdf(x + 1, y, z, f[1], weight);
                if (weight == 0) return 0;
                readTsdf(x + 1, y + 1, z, f[2], weight);
                if (weight == 0) return 0;
                readTsdf(x, y + 1, z, f[3], weight);
                if (weight == 0) return 0;
                readTsdf(x, y, z + 1, f[4], weight);
                if (weight == 0) return 0;
                readTsdf(x + 1, y, z + 1, f[5], weight);
                if (weight == 0) return 0;
                readTsdf(x + 1, y + 1, z + 1, f[6], weight);
                if (weight == 0) return 0;
                readTsdf(x, y + 1, z + 1, f[7], weight);
                if (weight == 0) return 0;

                // calculate flag indicating if each vertex is inside or outside isosurface
                int cubeindex;
                cubeindex = int(f[0] < isoValue());
                cubeindex += int(f[1] < isoValue()) * 2;
                cubeindex += int(f[2] < isoValue()) * 4;
                cubeindex += int(f[3] < isoValue()) * 8;
                cubeindex += int(f[4] < isoValue()) * 16;
                cubeindex += int(f[5] < isoValue()) * 32;
                cubeindex += int(f[6] < isoValue()) * 64;
                cubeindex += int(f[7] < isoValue()) * 128;

                return cubeindex;
            }
        };

        struct OccupiedVoxels : public CubeIndexEstimator
        {
            enum
            {
                CTA_SIZE_X = 32,
                CTA_SIZE_Y = 8,
                CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y,

                WARPS_COUNT = CTA_SIZE / Warp::WARP_SIZE
            };

            mutable int* voxels_indices;
            mutable int* vertices_number;
            int max_size;

            OccupiedVoxels(const TsdfVolume& _volume) : CubeIndexEstimator(_volume) {}

            __kf_device__
            void operator () () const
            {





                int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
                int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

                if (__all (x >= volume.dims.x) || __all (y >= volume.dims.y))
                  return;

                int ftid = Block::flattenedThreadId ();
                int warp_id = Warp::id();
                int lane_id = Warp::laneId();

                volatile __shared__ int warps_buffer[WARPS_COUNT];

                for (int z = 0; z < volume.dims.z- 1; z++)
                {
                    int numVerts = 0;;
                    if (x + 1 < volume.dims.x && y + 1 < volume.dims.y)
                    {
                        float field[8];
                        int cubeindex = computeCubeIndex (x, y, z, field);

                        // read number of vertices from texture
                        numVerts = (cubeindex == 0 || cubeindex == 255) ? 0 : tex1Dfetch (numVertsTex, cubeindex);
                    }

                    int total = __popc (__ballot (numVerts > 0));

                    if (total == 0)
                        continue;

                    if (lane_id == 0)
                    {
                        int old = atomicAdd (&global_count, total);
                        warps_buffer[warp_id] = old;
                    }
                    int old_global_voxels_count = warps_buffer[warp_id];
                    int offs = Warp::binaryExclScan (__ballot (numVerts > 0));

                    if (old_global_voxels_count + offs < max_size && numVerts > 0)
                    {
                        voxels_indices[old_global_voxels_count + offs] = volume.dims.y * volume.dims.x * z + volume.dims.x * y + x;
                        vertices_number[old_global_voxels_count + offs] = numVerts;
                    }

                    bool full = old_global_voxels_count + total >= max_size;

                    if (full)
                        break;

                } /* for(int z = 0; z < VOLUME_Z - 1; z++) */


                /////////////////////////
                // prepare for future scans
                if (ftid == 0)
                {
                    unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
                    unsigned int value = atomicInc (&blocks_done, total_blocks);

                    //last block
                    if (value == total_blocks - 1)
                    {
                        output_count = min (max_size, global_count);
                        blocks_done = 0;
                        global_count = 0;
                    }
                }
            }
        };
        __global__ void getOccupiedVoxelsKernel (const OccupiedVoxels ov) {ov();}
    } // namespace device
} // namespace kfusion

int kfusion::device::getOccupiedVoxels (const TsdfVolume& volume, DeviceArray2D<int>& occupied_voxels)
{
    OccupiedVoxels ov(volume);

    ov.voxels_indices = occupied_voxels.ptr (0);
    ov.vertices_number = occupied_voxels.ptr (1);
    ov.max_size = occupied_voxels.cols ();

    dim3 block (OccupiedVoxels::CTA_SIZE_X, OccupiedVoxels::CTA_SIZE_Y);
    dim3 grid (divUp (volume.dims.x, block.x), divUp (volume.dims.y, block.y));

    //cudaFuncSetCacheConfig(getOccupiedVoxelsKernel, cudaFuncCachePreferL1);
    //printFuncAttrib(getOccupiedVoxelsKernel);

    getOccupiedVoxelsKernel<<<grid, block>>>(ov);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());

    int size;
    cudaSafeCall ( cudaMemcpyFromSymbol (&size, output_count, sizeof(size)) );
    return size;
}

int kfusion::device::computeOffsetsAndTotalVertices (DeviceArray2D<int>& occupied_voxels)
{
    thrust::device_ptr<int> beg (occupied_voxels.ptr (1));
    thrust::device_ptr<int> end = beg + occupied_voxels.cols ();

    thrust::device_ptr<int> out (occupied_voxels.ptr (2));
    thrust::exclusive_scan (beg, end, out);

    int lastElement, lastScanElement;

    DeviceArray<int> last_elem (occupied_voxels.ptr(1) + occupied_voxels.cols () - 1, 1);
    DeviceArray<int> last_scan (occupied_voxels.ptr(2) + occupied_voxels.cols () - 1, 1);

    last_elem.download (&lastElement);
    last_scan.download (&lastScanElement);

    return lastElement + lastScanElement;
}

namespace kfusion
{
    namespace device
    {
        struct TrianglesGenerator : public CubeIndexEstimator
        {
            const int* occupied_voxels;
            const int* vertex_offsets;
            int voxels_count;
            float3 cell_size;
            Aff3f aff;

            mutable Point *output;

            TrianglesGenerator(const TsdfVolume& _volume) : CubeIndexEstimator(_volume) {}

            __kf_device__
            float3 getNodeCoo (int x, int y, int z) const
            {
                float3 coo = make_float3 (x, y, z);
                coo += 0.5f;                 //shift to volume cell center;

                coo.x *= cell_size.x;
                coo.y *= cell_size.y;
                coo.z *= cell_size.z;

                float3 worldp;
                worldp = aff * coo;

                return worldp;
            }

            __kf_device__
            float3 vertex_interp (float3 p0, float3 p1, float f0, float f1) const
            {
                float t = (isoValue() - f0) / (f1 - f0 + 1e-15f);
                float x = p0.x + t * (p1.x - p0.x);
                float y = p0.y + t * (p1.y - p0.y);
                float z = p0.z + t * (p1.z - p0.z);
                return make_float3 (x, y, z);
            }

            __kf_device__
            void store_point (float4 *ptr, int index, const float3& point) const
            {
                ptr[index] = make_float4 (point.x, point.y, point.z, 1.0f);
            }

            __kf_device__
            void operator() () const
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx >= voxels_count)
                    return;

                int voxel = occupied_voxels[idx];

                int z = voxel / (volume.dims.x * volume.dims.y);
                int y = (voxel - z * volume.dims.x * volume.dims.y) / volume.dims.x;
                int x = (voxel - z * volume.dims.x * volume.dims.y) - y * volume.dims.x;

                float f[8];
                int cubeindex = computeCubeIndex (x, y, z, f);

                // calculate cell vertex positions
                float3 v[8];
                v[0] = getNodeCoo (x, y, z);
                v[1] = getNodeCoo (x + 1, y, z);
                v[2] = getNodeCoo (x + 1, y + 1, z);
                v[3] = getNodeCoo (x, y + 1, z);
                v[4] = getNodeCoo (x, y, z + 1);
                v[5] = getNodeCoo (x + 1, y, z + 1);
                v[6] = getNodeCoo (x + 1, y + 1, z + 1);
                v[7] = getNodeCoo (x, y + 1, z + 1);

                // find the vertices where the surface intersects the cube
                // use shared memory to avoid using local
                __shared__ float3 vertlist[12][256];

                vertlist[0][threadIdx.x] = vertex_interp (v[0], v[1], f[0], f[1]);
                vertlist[1][threadIdx.x] = vertex_interp (v[1], v[2], f[1], f[2]);
                vertlist[2][threadIdx.x] = vertex_interp (v[2], v[3], f[2], f[3]);
                vertlist[3][threadIdx.x] = vertex_interp (v[3], v[0], f[3], f[0]);
                vertlist[4][threadIdx.x] = vertex_interp (v[4], v[5], f[4], f[5]);
                vertlist[5][threadIdx.x] = vertex_interp (v[5], v[6], f[5], f[6]);
                vertlist[6][threadIdx.x] = vertex_interp (v[6], v[7], f[6], f[7]);
                vertlist[7][threadIdx.x] = vertex_interp (v[7], v[4], f[7], f[4]);
                vertlist[8][threadIdx.x] = vertex_interp (v[0], v[4], f[0], f[4]);
                vertlist[9][threadIdx.x] = vertex_interp (v[1], v[5], f[1], f[5]);
                vertlist[10][threadIdx.x] = vertex_interp (v[2], v[6], f[2], f[6]);
                vertlist[11][threadIdx.x] = vertex_interp (v[3], v[7], f[3], f[7]);
                __syncthreads ();

                // output triangle vertices
                int numVerts = tex1Dfetch (numVertsTex, cubeindex);

                for (int i = 0; i < numVerts; i += 3)
                {
                    int index = vertex_offsets[idx] + i;

                    int v1 = tex1Dfetch (triTex, (cubeindex * 16) + i + 0);
                    int v2 = tex1Dfetch (triTex, (cubeindex * 16) + i + 1);
                    int v3 = tex1Dfetch (triTex, (cubeindex * 16) + i + 2);

                    store_point (output, index + 0, vertlist[v1][threadIdx.x]);
                    store_point (output, index + 1, vertlist[v2][threadIdx.x]);
                    store_point (output, index + 2, vertlist[v3][threadIdx.x]);
                }
            }
        };

        __global__ void trianglesGeneratorKernel (const TrianglesGenerator tg) {tg();}
    }
}

void kfusion::device::generateTriangles (const TsdfVolume& volume, const Aff3f& aff, const DeviceArray2D<int>& occupied_voxels, DeviceArray<Point>& output)
{
    const int block = 256;

    typedef TrianglesGenerator Tg;
    Tg tg(volume);

    tg.occupied_voxels = occupied_voxels.ptr (0);
    tg.vertex_offsets = occupied_voxels.ptr (2);
    tg.voxels_count = occupied_voxels.cols ();
    tg.cell_size = volume.voxel_size;
    tg.aff = aff;
    tg.output = output;

    trianglesGeneratorKernel<<<divUp (tg.voxels_count, block), block>>>(tg);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}