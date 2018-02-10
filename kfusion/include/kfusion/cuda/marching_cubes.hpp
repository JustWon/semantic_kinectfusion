/**
 *  @file   marching_cubes.hpp
 *  @brief
 *
 *  @author Gabriel Cuendet
 *  @date   12/07/2016
 *  Copyright (c) 2016 Gabriel Cuendet. All rights reserved.
 */

#include <kfusion/types.hpp>
#include <kfusion/cuda/device_array.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>

#ifndef __KFUSION_MARCHING_CUBES_HPP__
#define __KFUSION_MARCHING_CUBES_HPP__

namespace kfusion
{
    namespace cuda
    {
        /**
         * @class MarchingCubes
         * @brief MarchingCubes class
         */
        class KF_EXPORTS MarchingCubes
        {
        public:

#pragma mark -
#pragma mark Initialization

            /**
             * @name MarchingCubes
             * @fn MarchingCubes (void)
             * @brief Default constructor of the class
             */
            MarchingCubes (void);

            /**
             * @name ~MarchingCubes
             * @fn ~MarchingCubes(void)
             * @brief Default destructor of the class
             */
            ~MarchingCubes (void);

            /**
             * @name run
             * @fn DeviceArray<Point> run (const TsdfVolume& volume,
             *                             DeviceArray<Point>& triangles_buffer)
             * @brief runs marching cubes triangulation algorithm
             * @param[in] volume  The Tsdf volume to march
             * @param[in/out] triangles_buffer  A buffer for the extracted triangles
             * @return Array with the extracted triangles.
             *         The returned array points to 'triangles_buffer' data.
             */
            DeviceArray<Point> run(const TsdfVolume& volume,
                                   DeviceArray<Point>& triangles_buffer);

        private:
            /**< Edge table */
            DeviceArray<int> edgeTable_;
            /**< Triangles table corresponding to edgeTable_ */
            DeviceArray<int> triTable_;
            /**< Number of vertices corresponding to each case in edgeTable_ */
            DeviceArray<int> numVertsTable_;
            /**< Temporary buffer used by marching cubes:
             *     1st row stores occupied voxels ids
             *     2nd row stores number of vertices for the corresponding voxel
             *     3rd row stores points offsets (cumulative sum over the number of vertices */
            DeviceArray2D<int> occupied_voxels_buffer_;
        };
    }  // namespace cuda
}  // namespace kfusion

#endif //__KFUSION_MARCHING_CUBES_HPP__
