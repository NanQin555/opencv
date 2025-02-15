// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_WARPAFFINE_HPP_INCLUDED
#define OPENCV_HAL_RVV_WARPAFFINE_HPP_INCLUDED

#include <riscv_vector.h>
#include <cassert>
#include <iostream>
#include "opencv2/imgproc/hal/interface.h"
#include "opencv2/core/hal/interface.h"  

namespace cv { namespace cv_hal_rvv {
#undef cv_hal_remap32f
#define cv_hal_remap32f cv::cv_hal_rvv::remap32f

// todo 
#define OPENCV_HAL_IMPL_RISCV_CLIP(_Tp, x, num) \
inline __riscv_v##_Tp##x##num clip(v_##_Tp##x##num2 x, int a, int b) \
{ \
} \

static inline vint16m8_t clip(vint16m8_t x, int a, int b, int vl) {
    x = __riscv_vmax_vx_i16m8(x, a, vl); 
    x = __riscv_vmin_vx_i16m8(x, b-1, vl);
    return x;
}

vint16m8_t borderInterpolate(vint16m8_t p, int len, int borderType, int vl) {

    if( (unsigned)p < (unsigned)len )
        ;
    else if( borderType == CV_HAL_BORDER_REPLICATE ) {
        // p = p < 0 ? 0 : len - 1;
        auto mask = __riscv_vmslt_vx_i16m8_b2(p, 0, vl);
        auto vec_zero = __riscv_vmv_v_x_i16m8(0, vl);
        p = __riscv_vmerge_vxm_i16m8(vec_zero, len - 1, mask, vl);
    }
    else if( borderType == CV_HAL_BORDER_REFLECT || borderType == CV_HAL_BORDER_REFLECT_101 )
    {
        int delta = borderType == CV_HAL_BORDER_REFLECT_101;
        if( len == 1 )
            return 0;
        do
        {
            if( p < 0 ) 
            {
                // p = -p - 1 + delta;
                p = __riscv_vneg_v_i16m8(p, vl);
                p = __riscv_vadd_vx_i16m8(p, delta - 1, vl);
            }
            else 
            {
                // p = len - 1 - (p - len) - delta;
                p = __riscv_vneg_v_i16m8(p, vl);
                p = __riscv_vadd_vx_i16m8(p, 2 * len - 1 - delta, vl);
            }
        }
    }
    else if( borderType == CV_HAL_BORDER_WRAP )
    {
        std::assert(len > 0);
        // if( p < 0 ) 
        //     p -= ((p-len+1)/len)*len; 
        auto mask = __riscv_vmslt_vx_i16m8_b2(p, 0, vl);
        auto tmp = __riscv_vadd_vx_i16m8_m(mask, p, 1 - len, vl);
        tmp = __riscv_vdiv_vx_i16m8_m(mask, tmp, len, vl);
        tmp = __riscv_vmul_vx_i16m8_m(mask, tmp, len, vl);
        p = __riscv_vsub_vx_i16m8_m(mask, p, tmp, len, vl);
        // if( p >= len )
        //     p %= len;
        mask = __riscv_vmnot_m_b2(mask, vl);
        p = __riscv_vrem_vx_i16m8(mask, p, len, vl);
    }
    else if( borderType == CV_HAL_BORDER_CONSTANT )
        p = __riscv_vmv_v_x_i16m8(-1, vl); // p = -1; 
    else
        std::cerr << "Unknown/unsupported border type" << std::endl;
    return p;
}

// template<typename T, bool isRelative>
static void remapNearest(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height,
    uchar* dst_data, size_t dst_step, int dst_width, int dst_height, 
    short* mapx, size_t mapx_step, short* mapy, size_t mapy_step, 
    int border_type, const double border_value[4]) 
{
    // ssize, dsize -> src_width, src_height, dst_width, dst_height

    // only CV_8U
    if( (src_type & CV_MAT_DEPTH_MASK) != CV_8U )
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const int cn = CV_MAT_CN(src_type);
    const uchar* S0 = src_data;
    uchar cval[CV_CN_MAX];

    size_t sstep = _src.step/sizeof(S0[0]);
    size_t mstep = mapx_step/sizeof(float);

    for(int k = 0; k < cn; k++ ) 
        cval[k] = saturate_cast<T>(_borderValue[k & 3]);
    
    for(int y = 0; y < dst_height; ++y) 
    {
        uchar *D = dst_data;
        int vl;
        if(cn == 1) 
        {
            for(int x = 0; x < dst_width; x += vl) 
            {
                vl = __riscv_vsetvl_e8m4(dst_height - x);
                // map type is 16SC1
                auto mapx_vector = __riscv_vle16_v_i16m8(mapx, vl);
                auto mapy_vector = __riscv_vle16_v_i16m8(mapy, vl); 
                auto mapx_mask = __riscv_vmslt_vx_i16m8_b2(mapx_vector, (unsigned)src_width, vl);
                auto mapy_mask = __riscv_vmslt_vx_i16m8_b2(mapy_vector, (unsigned)src_height, vl);
                auto map_mask = __riscv_vmand_mm_b2(mapx_mask, mapy_mask, vl);
                
                // index size ?
                auto index_vector = __riscv_vmul_vx_i16m8(mapy_vector, sstep, vl);
                index_vector = __riscv_vadd_vv_i16m8(index_vector, mapx_vector, vl);

                // valid
                auto S0_vector = __riscv_vloxei8_v_u8m4(S0, index_vector, vl);
                __riscv_vse8_v_u8m4(map_mask, D+x, S0_vector, vl);
                
                if ( __riscv_vcpop_m_b2(map_mask, vl) != vl) 
                {
                    // invalid
                    map_mask = __riscv_vmnot_m_b2(map_mask, vl);
                    if(border_type == CV_HAL_BORDER_REFLECT) 
                    {
                        mapx_vector = clip(mapx_vector, 0, src_width, vl);
                        mapy_vector = clip(mapy_vector, 0, src_height, vl);
                        index_vector = __riscv_vmul_vx_i16m8(mapy_vector, sstep, vl);
                        index_vector = __riscv_vadd_vv_i16m8(index_vector, mapx_vector, vl);    
                        S0_vector = __riscv_vloxei32_v_u8m4(S0, index_vector, vl);
                        __riscv_vse8_v_u8m4_m(map_mask, D+x, S0_vector, vl);
                    }
                    else if(border_type == CV_HAL_BORDER_CONSTANT) 
                    {
                        __riscv_vse8_vx_u8m4_m(map_mask, D+x, cval[0], vl);
                    }
                    else if(border_type != CV_HAL_BORDER_TRANSPARENT) 
                    {
                        mapx_vector = borderInterpolate(mapx_vector, src_width, border_type, vl);
                        mapy_vector = borderInterpolate(mapy_vector, src_height, border_type, vl);
                        index_vector = __riscv_vmul_vx_i16m8(mapy_vector, sstep, vl);
                        index_vector = __riscv_vadd_vv_i16m8(index_vector, mapx_vector, vl);    
                        S0_vector = __riscv_vloxei32_v_u8m4(S0, index_vector, vl);
                        __riscv_vse8_v_u8m4_m(map_mask, D+x, S0_vector, vl);
                    }
                }
            }
        }
        else 
        {
            for(int x = 0; x < dst_width; x += vl, D += cn * vl) 
            {
                vl = __riscv_vsetvl_e8m4(dst_height - x);

                auto mapx_vector = __riscv_vle8_v_i16m8(mapx, vl);
                auto mapy_vector = __riscv_vle8_v_i16m8(mapy, vl); 
                auto mapx_mask = __riscv_vmslt_vx_i16m8_b2(mapx_vector, src_width, vl);
                auto mapy_mask = __riscv_vmslt_vx_i16m8_b2(mapy_vector, src_height, vl);
                auto map_mask = __riscv_vand_vv(mapx_mask, mapy_mask, vl);
                
                // valid
                auto index_vector = __riscv_vmv_v_x_i16m8(0, vl);
                index_vector = __riscv_vmul_vx_i16m8_tumu(map_mask, index_vector, mapy_vector, sstep, vl);
                auto index_x = __riscv_vmul_vx_i16m8_tumu(map_mask, mapx_vector, mapx_vector, cn, vl);
                index_vector = __riscv_vadd_vv_i16m8_tumu(map_mask, index_vector, index_vector, index_x, vl);

                // invalid
                map_mask = __riscv_vmnot_m_b2(map_mask, vl);
                if(border_type != CV_HAL_BORDER_TRANSPARENT) 
                {
                    if(border_type == CV_HAL_BORDER_REPLICATE) 
                    {
                        mapx_vector = clip(mapx_vector, 0, src_width, vl);
                        mapy_vector = clip(mapy_vector, 0, src_height);
                    }
                    else if(border_type == CV_HAL_BORDER_CONSTANT)
                    {
                        // S = &cval[0];
                        auto S0_vector = __riscv_vmv_v_x_u8m4(0, vl);
                        for(int k = 0; k <  cn; ++k)
                        {
                            S0_vector = __riscv_vle8_v_u8m4(&cval[0] + cn - 1, vl);
                            __riscv_vsse8_v_u8m4(D + cn - 1, sizeof(uchar)*cn, S0_vector, vl);
                        }
                        continue;
                    }
                    else 
                    {
                        mapx_vector = borderInterpolate(mapx_vector, src_width, border_type, vl);
                        mapy_vector = borderInterpolate(mapy_step, src_height, border_type, vl);
                    }
                    index_vector = __riscv_vmul_vx_i16m8_tumu(map_mask, index_vector, mapy_vector, sstep, vl);
                    index_x = __riscv_vmul_vx_i16m8_tumu(map_mask, index_x, mapx_vector, cn, vl);
                    index_vector = __riscv_vadd_vv_i16m8_tumu(map_mask, index_vector, index_vector, index_x, vl);
                }
                // for(int k = 0; k < cn; k++ ) 
                //     D[k] = S[k];
                auto S0_vector = __riscv_vmv_v_x_u8m4(0, vl);
                for(int k = 0; k <  cn; ++k)
                {
                    S0_vector = __riscv_vloxei32_v_u8m4(S0 + cn - 1, index_vector, vl);
                    __riscv_vsse8_v_u8m4(D + cn - 1, sizeof(uchar)*cn, S0_vector, vl);
                }
            }

        }
    }
}

#undef cv_hal_remap16s
#define cv_hal_remap16s cv::cv_hal_rvv::remap16s
// mapx_type = CV_16SC2
static int remap16s(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height,
    uchar* dst_data, size_t dst_step, int dst_width, int dst_height, 
    short* mapx, size_t mapx_step, int mapy_type, short* mapy, size_t mapy_step, 
    int interpolation, int border_type, const double border_value[4]) 
{
    const bool isRelative = ((interpolation & CV_HAL_WARP_RELATIVE_MAP) != 0);
    interpolation &= ~CV_HAL_WARP_RELATIVE_MAP;

    if( interpolation != CV_HAL_INTER_NEAREST )
    return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if( interpolation == CV_HAL_INTER_AREA )
        interpolation = CV_HAL_INTER_LINEAR;

    if( (mapy_type != CV_16SC1) )
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // mapxy -> mapx, mapy
    int vl, step = mapx_step/sizeof(mapx[0]);
    for(int x = 0; x < mapx_step; x += vl)
    {
        vl = __riscv_vsetvl_i16m8(mapx_step - x);
        auto vx = __riscv_vlse16_v_i16m8(mapx, sizeof(short)*2, vl);
        auto vy = __riscv_vlse16_v_i16m8(mapx + 1, sizeof(short)*2, vl);
        __riscv_vse16_v_i16m8(mapx, vx, vl);
        __riscv_vse16_v_i16m8(mapy, vy, vl);
    }

    // run RemapFunc
    int depth = CV_MAT_DEPTH(src_type);
    switch(depth) 
    {
        case: CV_8U
        remapNearest(src_type, src_data, src_step, src_width, src_height,
                     dst_data, dst_step, dst_width, dst_height, 
                     mapx, mapx_step/2, mapy, mapx_step/2, border_type, border_value[4]);
    }
}

} // cv_hal_rvv::
} // cv::

#endif