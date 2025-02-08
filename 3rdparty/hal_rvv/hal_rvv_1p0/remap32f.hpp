// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_WARPAFFINE_HPP_INCLUDED
#define OPENCV_HAL_RVV_WARPAFFINE_HPP_INCLUDED

#include <riscv_vector.h>
#include "opencv2/imgproc/hal/interface.h"
#include "opencv2/core/hal/interface.h"
namespace cv { namespace cv_hal_rvv {

#undef cv_hal_remap32f
#define cv_hal_remap32f cv::cv_hal_rvv::remap32f

template<typename T, bool isRelative>
static void remapNearest(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height,
    uchar* dst_data, size_t dst_step, int dst_width, int dst_height, 
    // float* mapx, size_t mapx_step,
    // float* mapy, size_t mapy_step, 
    float* mapxy, size_t mapxy_step,
    int interpolation, int border_type, const double border_value[4]) {
    
    const int cn = CV_MAT_CN(src_type);
    const T* S0 = src_data;
    T cval[CV_CN_MAX];
    size_t sstep = _src.step/sizeof(S0[0]);

    for(int k = 0; k < cn; k++ )
        cval[k] = saturate_cast<T>(_borderValue[k & 3]);
    
    int vl;
    for(int dy = 0; dy < dst_height; ++dy) {
        T* D = dst_data;
        const short* XY = einterpret_cast<const short*>mapxy + dy;
        vl = __riscv_vsetvl_e32m8(dst_height - dy);
    }
}


// 32fc1 
// map1 -> x, map2 -> y
// if interpolation = INTER_NEAREST -> nnfunc != 0
// planar_input = 1
static int remap32f(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height,
    uchar* dst_data, size_t dst_step, int dst_width, int dst_height, float* mapx, size_t mapx_step,
    float* mapy, size_t mapy_step, int interpolation, int border_type, const double border_value[4]) {
    
    const bool isRelative = ((interpolation & CV_HAL_WARP_RELATIVE_MAP) != 0);
    interpolation &= ~CV_HAL_WARP_RELATIVE_MAP;

    if( interpolation == CV_HAL_INTER_AREA )
        interpolation = CV_HAL_INTER_LINEAR;

    int depth = CV_MAT_DEPTH(src_type);
    // To Get RemapFunc

    // mapx mapy -> mapxy

    // run RemapFunc
//-----------------------------------------------------------------//
    int x, y, x1, y1;
    for(; x < )
    const int buf_size = 1 << 14;
    int brows0 = std::min(128, dst->rows), map_depth = m1->depth();
    int bcols0 = std::min(buf_size/brows0, dst->cols);
    brows0 = std::min(buf_size/bcols0, dst->rows);

    Mat _bufxy(brows0, bcols0, CV_16SC2), _bufa;
    if( !nnfunc )
        _bufa.create(brows0, bcols0, CV_16UC1);

    for( y = range.start; y < range.end; y += brows0 )
    {
        for( x = 0; x < dst->cols; x += bcols0 )
        {
            int brows = std::min(brows0, range.end - y);
            int bcols = std::min(bcols0, dst->cols - x);
            Mat dpart(*dst, Rect(x, y, bcols, brows));
            Mat bufxy(_bufxy, Rect(0, 0, bcols, brows));
            // todo
            for( y1 = 0; y1 < brows; y1++ )
            {
                short* XY = bufxy.ptr<short>(y1);
                const float* sX = m1->ptr<float>(y+y1) + x;
                const float* sY = m2->ptr<float>(y+y1) + x;
                x1 = 0;
                #if CV_SIMD128
                {
                    int span = VTraits<v_float32x4>::vlanes();
                    for( ; x1 <= bcols - span * 2; x1 += span * 2 )
                    {
                        v_int32x4 ix0 = v_round(v_load(sX + x1));
                        v_int32x4 iy0 = v_round(v_load(sY + x1));
                        v_int32x4 ix1 = v_round(v_load(sX + x1 + span));
                        v_int32x4 iy1 = v_round(v_load(sY + x1 + span));

                        v_int16x8 dx, dy;
                        dx = v_pack(ix0, ix1);
                        dy = v_pack(iy0, iy1);
                        v_store_interleave(XY + x1 * 2, dx, dy);
                    }
                }
                #endif
                for( ; x1 < bcols; x1++ )
                {
                    XY[x1*2] = saturate_cast<short>(sX[x1]);
                    XY[x1*2+1] = saturate_cast<short>(sY[x1]);
                }
            }

            // nnfunc( *src, dpart, bufxy, borderType, borderValue, Point(x, y) );
            Size ssize = _src.size(), dsize = _dst.size();
            const int cn = _src.channels();
            const T* S0 = _src.ptr<T>();
            T cval[CV_CN_MAX];
            size_t sstep = _src.step/sizeof(S0[0]);

            for(int k = 0; k < cn; k++ )
                cval[k] = saturate_cast<T>(_borderValue[k & 3]);

            unsigned width1 = ssize.width, height1 = ssize.height;

            if( _dst.isContinuous() && _xy.isContinuous() && !isRelative )
            {
                dsize.width *= dsize.height;
                dsize.height = 1;
            }

            for(int dy = 0; dy < dsize.height; dy++ )
            {
                T* D = _dst.ptr<T>(dy);
                const short* XY = _xy.ptr<short>(dy);
                const int off_y = isRelative ? (_offset.y+dy) : 0;
                if( cn == 1 )
                {
                    for(int dx = 0; dx < dsize.width; dx++ )
                    {
                        const int off_x = isRelative ? (_offset.x+dx) : 0;
                        int sx = XY[dx*2]+off_x, sy = XY[dx*2+1]+off_y;
                        if( (unsigned)sx < width1 && (unsigned)sy < height1 )
                            D[dx] = S0[sy*sstep + sx];
                        else
                        {
                            if( borderType == BORDER_REPLICATE )
                            {
                                sx = clip(sx, 0, ssize.width);
                                sy = clip(sy, 0, ssize.height);
                                D[dx] = S0[sy*sstep + sx];
                            }
                            else if( borderType == BORDER_CONSTANT )
                                D[dx] = cval[0];
                            else if( borderType != BORDER_TRANSPARENT )
                            {
                                sx = borderInterpolate(sx, ssize.width, borderType);
                                sy = borderInterpolate(sy, ssize.height, borderType);
                                D[dx] = S0[sy*sstep + sx];
                            }
                        }
                    }
                }
                else
                {
                    for(int dx = 0; dx < dsize.width; dx++, D += cn )
                    {
                        const int off_x = isRelative ? (_offset.x+dx) : 0;
                        int sx = XY[dx*2]+off_x, sy = XY[dx*2+1]+off_y;
                        const T *S;
                        if( (unsigned)sx < width1 && (unsigned)sy < height1 )
                        {
                            if( cn == 3 )
                            {
                                S = S0 + sy*sstep + sx*3;
                                D[0] = S[0], D[1] = S[1], D[2] = S[2];
                            }
                            else if( cn == 4 )
                            {
                                S = S0 + sy*sstep + sx*4;
                                D[0] = S[0], D[1] = S[1], D[2] = S[2], D[3] = S[3];
                            }
                            else
                            {
                                S = S0 + sy*sstep + sx*cn;
                                for(int k = 0; k < cn; k++ )
                                    D[k] = S[k];
                            }
                        }
                        else if( borderType != BORDER_TRANSPARENT )
                        {
                            if( borderType == BORDER_REPLICATE )
                            {
                                sx = clip(sx, 0, ssize.width);
                                sy = clip(sy, 0, ssize.height);
                                S = S0 + sy*sstep + sx*cn;
                            }
                            else if( borderType == BORDER_CONSTANT )
                                S = &cval[0];
                            else
                            {
                                sx = borderInterpolate(sx, ssize.width, borderType);
                                sy = borderInterpolate(sy, ssize.height, borderType);
                                S = S0 + sy*sstep + sx*cn;
                            }
                            for(int k = 0; k < cn; k++ )
                                D[k] = S[k];
                        }
                    }
                }
            }
            continue;
        }
    }
}




} // cv_hal_rvv::
} // cv::

#endif