// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_MEANSTDDEV_HPP_INCLUDED
#define OPENCV_HAL_RVV_MEANSTDDEV_HPP_INCLUDED
#include <riscv_vector.h>
#include <iostream>

namespace cv { namespace cv_hal_rvv {
#undef cv_hal_meanStdDev
#define cv_hal_meanStdDev cv::cv_hal_rvv::meanStdDev

inline int meanStdDev_8uc1(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val, uchar* mask, size_t mask_step);
inline int meanStdDev_8uc4(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val, uchar* mask, size_t mask_step);
inline int meanStdDev_32fc1(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val, uchar* mask, size_t mask_step);

inline int meanStdDev(const uchar* src_data, size_t src_step, int width, int height,
                             int src_type, double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
    switch (src_type) {
        case CV_8UC1:
            return meanStdDev_8uc1(src_data, src_step, width, height, mean_val, stddev_val, mask, mask_step);
        // case CV_8UC4:
        //     return meanStdDev_8uc4(src_data, src_step, width, height, mean_val, stddev_val, mask, mask_step);
        // case CV_32FC1:
        //     return meanStdDev_32fc1(src_data, src_step, width, height, mean_val, stddev_val, mask, mask_step);
        default:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
}


// inline int meanStdDev_8uc1(const uchar* src_data, size_t src_step, int width, int height,
//                             double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
//     // size_t loop_unroll = 4;
//     // size_t loop_rem = height % loop_unroll; 
//     // initialize variables
//     size_t total_count = 0;
//     size_t vl = __riscv_vsetvlmax_e8m1();
//     uint64_t sum = 0;
//     std::cout << "height: " << height << " width: " << width << " src_step: " << src_step << " mask_step: " << mask_step << std::endl;
//     vuint64m4_t vec_sum = __riscv_vmv_v_x_u64m4(0, vl);
//     vuint64m1_t vec_sqsum = __riscv_vmv_v_x_u64m1(0, vl);
//     vuint16m1_t vec_s = __riscv_vmv_v_x_u16m1(0, vl);
//     vuint16m1_t u16_zero = __riscv_vmv_v_x_u16m1(0, vl);
//     vuint32m1_t temp_sqsum = __riscv_vmv_v_x_u32m1(0, vl);
//     size_t i = 0;
//     // for ( ; i < height - loop_rem; i+=4) {
//     //     const uchar* src_row1 = src_data + i * src_step;
//     //     const uchar* src_row2 = src_data + ( i + 1 ) * src_step;
//     //     const uchar* src_row3 = src_data + ( i + 2 ) * src_step;
//     //     const uchar* src_row4 = src_data + ( i + 3 ) * src_step;
//     //     const uchar* mask_row1 = mask ? (mask + i * mask_step) : nullptr;
//     //     const uchar* mask_row2 = mask ? (mask + ( i + 1 ) * mask_step) : nullptr;
//     //     const uchar* mask_row3 = mask ? (mask + ( i + 2 ) * mask_step) : nullptr;
//     //     const uchar* mask_row4 = mask ? (mask + ( i + 3 ) * mask_step) : nullptr;
//     //     for ( ; j < width; ) {
//     //         vl = __riscv_vsetvl_e8m1(width-j); // tail elements
//     //         vuint16m2_t pixel_squared_vector1;
//     //         vuint16m2_t pixel_squared_vector2;
//     //         vuint16m2_t pixel_squared_vector3;
//     //         vuint16m2_t pixel_squared_vector4;
//     //         // Load src[row][i .. i+vl]
//     //         vuint8m1_t pixel_vector1 = __riscv_vle8_v_u8m1(src_row1 + j, vl);
//     //         vuint8m1_t pixel_vector2 = __riscv_vle8_v_u8m1(src_row2 + j, vl);
//     //         vuint8m1_t pixel_vector3 = __riscv_vle8_v_u8m1(src_row3 + j, vl);
//     //         vuint8m1_t pixel_vector4 = __riscv_vle8_v_u8m1(src_row4 + j, vl);
//     //         if(mask) {
//     //             // Load mask[row][i .. i+vl]
//     //             vbool8_t mask_vector1 = __riscv_vlm_v_b8(mask_row1 + j, vl);
//     //             vbool8_t mask_vector2 = __riscv_vlm_v_b8(mask_row2 + j, vl);
//     //             vbool8_t mask_vector3 = __riscv_vlm_v_b8(mask_row3 + j, vl);
//     //             vbool8_t mask_vector4 = __riscv_vlm_v_b8(mask_row4 + j, vl);
//     //             // vec_s[0] <- sum(vec_s[0] , pixel_vector[*]) , if not masked
//     //             vec_s = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector1, pixel_vector1, vec_s, vl);
//     //             vec_s = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector2, pixel_vector2, vec_s, vl);
//     //             vec_s = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector3, pixel_vector3, vec_s, vl);
//     //             vec_s = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector4, pixel_vector4, vec_s, vl);
//     //             if(stddev_val) {
//     //                 // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*] , if not masked
//     //                 pixel_squared_vector1 = __riscv_vwmulu_vv_u16m2_m(mask_vector1, pixel_vector1, pixel_vector1, vl);
//     //                 pixel_squared_vector2 = __riscv_vwmulu_vv_u16m2_m(mask_vector2, pixel_vector2, pixel_vector2, vl);
//     //                 pixel_squared_vector3 = __riscv_vwmulu_vv_u16m2_m(mask_vector3, pixel_vector3, pixel_vector3, vl);
//     //                 pixel_squared_vector4 = __riscv_vwmulu_vv_u16m2_m(mask_vector4, pixel_vector4, pixel_vector4, vl);
//     //                 // sqsum[0] <- sum(temp_sqsum[0] , pixel_squared_vector[*]) , if not masked
//     //                 temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector1, pixel_squared_vector1, temp_sqsum, vl);
//     //                 temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector2, pixel_squared_vector2, temp_sqsum, vl);
//     //                 temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector3, pixel_squared_vector3, temp_sqsum, vl);
//     //                 temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector4, pixel_squared_vector4, temp_sqsum, vl);
//     //             }
//     //         }
//     //         else {
//     //             // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*])
//     //             vec_s = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector1, vec_s, vl);
//     //             vec_s = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector2, vec_s, vl);
//     //             vec_s = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector3, vec_s, vl);
//     //             vec_s = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector4, vec_s, vl);
//     //             if(stddev_val) {
//     //                 // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*]
//     //                 pixel_squared_vector1 = __riscv_vwmulu_vv_u16m2(pixel_vector1, pixel_vector1, vl);
//     //                 pixel_squared_vector2 = __riscv_vwmulu_vv_u16m2(pixel_vector2, pixel_vector2, vl);
//     //                 pixel_squared_vector3 = __riscv_vwmulu_vv_u16m2(pixel_vector3, pixel_vector3, vl);
//     //                 pixel_squared_vector4 = __riscv_vwmulu_vv_u16m2(pixel_vector4, pixel_vector4, vl);
//     //                 // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*])
//     //                 temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector1, temp_sqsum, vl);
//     //                 temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector2, temp_sqsum, vl);
//     //                 temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector3, temp_sqsum, vl);
//     //                 temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector4, temp_sqsum, vl);
//     //             }
//     //         }
//     //         // vuint64m1_t <- vuint16m1_t
//     //         vuint64m1_t temp_sum = __riscv_vreinterpret_u64m1(vec_s);
//     //         // vec_sum[0] = sum( temp_sum , vec_sum)
//     //         vec_sum = __riscv_vadd_vv_u64m1(temp_sum, vec_sum, vl);
//     //         if (stddev_val) {
//     //             temp_sum = __riscv_vreinterpret_u64m1(temp_sqsum);
//     //             vec_sqsum = __riscv_vadd_vv_u64m1(temp_sum, vec_sqsum, vl);
//     //         }
//     //         total_count += vl;
//     //         j += vl;
//     //     }
//     // }
//     for (i ; i < (size_t)height; ++i) {
//         std::cout << "i:" << i << std::endl;
//         const uchar* src_row = src_data + i * src_step;
//         const uchar* mask_row = mask ? (mask + i * mask_step) : nullptr;
//         for (size_t j = 0 ; j < (size_t)width; j+=vl, total_count+=vl) {
//             vl = __riscv_vsetvl_e8m1(width-j); // tail elements
//             std::cout << "j:" << j << " vl: " << vl << std::endl;
//             // Load src[row][i .. i+vl]
//             vuint8m1_t pixel_vector = __riscv_vle8_v_u8m1(src_row + j, vl);
//             uint8_t data[vl];
//             __riscv_vse8_v_u8m1(data, pixel_vector, vl);
//             std::cout << "pixel_vector: ";
//             for(size_t k=0; k<vl; k++) {
//                 std::cout  << (int)data[k] << " ";
//             }
//             std::cout  << std::endl;
//             // vuint16m2_t pixel_squared_vector = __riscv_vmv_v_x_u16m2(0, vl);
//             if(mask) {
//                 // Load mask[row][i .. i+vl]              
//                 vbool8_t mask_vector = __riscv_vlm_v_b8(mask_row + j, vl);
//                 uint8_t maskdata[vl];
//                 __riscv_vsm_v_b8(maskdata, mask_vector, vl);
//                 std::cout << "mask_vector: ";
//                 for(size_t k=0; k<vl; k++) {
//                     std::cout  << (bool)maskdata[k] << " ";
//                 }
//                 std::cout  << std::endl;
//                 // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*]) , if not masked
//                 vec_s = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector, pixel_vector, u16_zero, vl);
//                 // if(stddev_val) {
//                 //     // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*] , if not masked
//                 //     pixel_squared_vector = __riscv_vwmulu_vv_u16m2_m(mask_vector, pixel_vector, pixel_vector, vl);
//                 //     // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*]) , if not masked
//                 //     temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector, pixel_squared_vector, temp_sqsum, vl);
//                 // }
//             }
//             else {
//                 // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*])
//                 vec_s = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector, u16_zero, vl);
//                 // if(stddev_val) {
//                 //     // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*]
//                 //     pixel_squared_vector = __riscv_vwmulu_vv_u16m2(pixel_vector, pixel_vector, vl);
//                 //     // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*])
//                 //     temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector, temp_sqsum, vl);
//                 // }
//             }
//             // vuint64m1_t <- vuint16m1_t
//             // vuint64m4_t temp_sum = __riscv_vzext_vf4_u64m4(vec_s, vl);
//             // uint64_t temp_sum_1 = __riscv_vmv_x_s_u64m4_u64(temp_sum);
//             auto temp_sum = __riscv_vmv_x_s_u16m1_u16(vec_s);
//             // vec_sum[0] = sum( temp_sum , vec_sum)
//             sum += static_cast<uint64_t>(temp_sum);
//             std::cout << "temp_sum: " << temp_sum << " sum: " << sum << " total_count: " << total_count+vl << std::endl;
//             // sum += temp_sum_1;
//             // vec_sum = __riscv_vadd_vv_u64m4(temp_sum, vec_sum, vl);
//             // if (stddev_val) {
//             //     // temp_sum = __riscv_vreinterpret_u64m1(temp_sqsum);
//             //     temp_sum = __riscv_vzext_vf4_u64m1(temp_sqsum);
//             //     vec_sqsum = __riscv_vadd_vv_u64m1(temp_sum, vec_sqsum, vl);
//             // }
//         }
//     }
//     if (total_count == 0)
//     {
//         if (mean_val) *mean_val = 0.0;
//         if (stddev_val) *stddev_val = 0.0;
//         return CV_HAL_ERROR_OK;
//     }
//     // Return values
//     // vfloat64m1_t float_sum = __riscv_vfcvt_f_xu_v_f64m1(vec_sum, vl);
//     // double dsum = __riscv_vfmv_f_s_f64m1_f64(float_sum);
//     std::cout << "sum: " << sum << " total_count: " << total_count << std::endl;
//     double mean = static_cast<double>(sum) / total_count;
//     if (mean_val) {
//         *mean_val = mean;
//     }
//     // if (stddev_val) {
//     //     vfloat64m1_t float_sqsum = __riscv_vfcvt_f_xu_v_f64m1(vec_sqsum, vl);
//     //     double sqsum = __riscv_vfmv_f_s_f64m1_f64(float_sqsum);
//     //     double variance = std::max((sqsum / total_count) - (mean * mean), 0.0);
//     //     double stddev = std::sqrt(variance);
//     //     *stddev_val = stddev;
//     // }
//     return CV_HAL_ERROR_OK;
// }

inline int meanStdDev_8uc1(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {

    // if(mask) {
    //     std::cout << "mask: " ;
    //     for(size_t n =0; n<64; ++n) {
    //         std::cout << (bool)mask[n] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    size_t loop_unroll = 4;
    size_t loop_rem = height % loop_unroll;
    // initialize variables
    size_t total_count = 0;
    uint64_t sum = 0, sqsum = 0;

    std::cout << "height: " << height << " width: " << width << " src_step: " << src_step << " mask_step: " << mask_step << std::endl;

    size_t vl = __riscv_vsetvlmax_e8m1();
    vuint16m1_t vec_sum = __riscv_vmv_v_x_u16m1(0, vl);
    vuint32m1_t vec_sqsum = __riscv_vmv_v_x_u32m1(0, vl);
    size_t i = 0;
    for ( ; i < (size_t)height - loop_rem; i+=4) {
        const uchar* src_row1 = src_data + i * src_step;
        const uchar* src_row2 = src_data + ( i + 1 ) * src_step;
        const uchar* src_row3 = src_data + ( i + 2 ) * src_step;
        const uchar* src_row4 = src_data + ( i + 3 ) * src_step;
        const uchar* mask_row1 = mask ? (mask + i * mask_step) : nullptr;
        const uchar* mask_row2 = mask ? (mask + ( i + 1 ) * mask_step) : nullptr;
        const uchar* mask_row3 = mask ? (mask + ( i + 2 ) * mask_step) : nullptr;
        const uchar* mask_row4 = mask ? (mask + ( i + 3 ) * mask_step) : nullptr;
        size_t j = 0 ;
        vl = __riscv_vsetvl_e8m1(width-j);
        for ( ; j < width; j+=vl) {
            vl = __riscv_vsetvl_e8m1(width-j); // tail elements
            vec_sum = __riscv_vmv_v_x_u16m1(0, vl);
            if(stddev_val)
                vec_sqsum = __riscv_vmv_v_x_u32m1(0, vl);
            // Load src[row][i .. i+vl]
            vuint8m1_t pixel_vector1 = __riscv_vle8_v_u8m1(src_row1 + j, vl);
            vuint8m1_t pixel_vector2 = __riscv_vle8_v_u8m1(src_row2 + j, vl);
            vuint8m1_t pixel_vector3 = __riscv_vle8_v_u8m1(src_row3 + j, vl);
            vuint8m1_t pixel_vector4 = __riscv_vle8_v_u8m1(src_row4 + j, vl);
            if(mask) {
                // Load mask[row][i .. i+vl]
                vuint8m1_t mask_load1 = __riscv_vle8_v_u8m1(mask_row1 + j, vl);
                vuint8m1_t mask_load2 = __riscv_vle8_v_u8m1(mask_row2 + j, vl);
                vuint8m1_t mask_load3 = __riscv_vle8_v_u8m1(mask_row3 + j, vl);
                vuint8m1_t mask_load4 = __riscv_vle8_v_u8m1(mask_row4 + j, vl);
                vbool8_t mask_vector1 =  __riscv_vmsne_vx_u8m1_b8(mask_load1, 0, vl);
                vbool8_t mask_vector2 =  __riscv_vmsne_vx_u8m1_b8(mask_load2, 0, vl);
                vbool8_t mask_vector3 =  __riscv_vmsne_vx_u8m1_b8(mask_load3, 0, vl);
                vbool8_t mask_vector4 =  __riscv_vmsne_vx_u8m1_b8(mask_load4, 0, vl);

                // vec_s[0] <- sum(vec_s[0] , pixel_vector[*]) , if not masked
                vec_sum = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector1, pixel_vector1, vec_sum, vl);
                vec_sum = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector2, pixel_vector2, vec_sum, vl);
                vec_sum = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector3, pixel_vector3, vec_sum, vl);
                vec_sum = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector4, pixel_vector4, vec_sum, vl);
                if(stddev_val) {
                    // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*] , if not masked
                    auto pixel_squared_vector1 = __riscv_vwmulu_vv_u16m2_m(mask_vector1, pixel_vector1, pixel_vector1, vl);
                    auto pixel_squared_vector2 = __riscv_vwmulu_vv_u16m2_m(mask_vector2, pixel_vector2, pixel_vector2, vl);
                    auto pixel_squared_vector3 = __riscv_vwmulu_vv_u16m2_m(mask_vector3, pixel_vector3, pixel_vector3, vl);
                    auto pixel_squared_vector4 = __riscv_vwmulu_vv_u16m2_m(mask_vector4, pixel_vector4, pixel_vector4, vl);
                    // sqsum[0] <- sum(temp_sqsum[0] , pixel_squared_vector[*]) , if not masked
                    vec_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector1, pixel_squared_vector1, vec_sqsum, vl);
                    vec_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector2, pixel_squared_vector2, vec_sqsum, vl);
                    vec_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector3, pixel_squared_vector3, vec_sqsum, vl);
                    vec_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector4, pixel_squared_vector4, vec_sqsum, vl);
                }
                total_count +=  __riscv_vcpop_m_b8(mask_vector1, vl);
                total_count +=  __riscv_vcpop_m_b8(mask_vector2, vl);
                total_count +=  __riscv_vcpop_m_b8(mask_vector3, vl);
                total_count +=  __riscv_vcpop_m_b8(mask_vector4, vl);
            }
            else {
                // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*])
                vec_sum = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector1, vec_sum, vl);
                vec_sum = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector2, vec_sum, vl);
                vec_sum = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector3, vec_sum, vl);
                vec_sum = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector4, vec_sum, vl);
                if(stddev_val) {
                    // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*]
                    auto pixel_squared_vector1 = __riscv_vwmulu_vv_u16m2(pixel_vector1, pixel_vector1, vl);
                    auto pixel_squared_vector2 = __riscv_vwmulu_vv_u16m2(pixel_vector2, pixel_vector2, vl);
                    auto pixel_squared_vector3 = __riscv_vwmulu_vv_u16m2(pixel_vector3, pixel_vector3, vl);
                    auto pixel_squared_vector4 = __riscv_vwmulu_vv_u16m2(pixel_vector4, pixel_vector4, vl);
                    // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*])
                    vec_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector1, vec_sqsum, vl);
                    vec_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector2, vec_sqsum, vl);
                    vec_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector3, vec_sqsum, vl);
                    vec_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector4, vec_sqsum, vl);
                }
                total_count += vl * 4;
            }
            auto temp_sum = __riscv_vmv_x_s_u16m1_u16(vec_sum);
            sum += static_cast<uint64_t>(temp_sum);
            if (stddev_val) {
                auto temp_sqsum = __riscv_vmv_x_s_u32m1_u32(vec_sqsum);
                sqsum += static_cast<uint64_t>(temp_sqsum);
            }
            std::cout << "temp_sum: " << temp_sum << " sum: " << sum << " total_count: " << total_count << std::endl;
        }
    }
    vec_sum = __riscv_vmv_v_x_u16m1(0, vl);
    vec_sqsum = __riscv_vmv_v_x_u32m1(0, vl);
    vuint16m1_t u16_zero = __riscv_vmv_v_x_u16m1(0, vl);
    vuint32m1_t u32_zero = __riscv_vmv_v_x_u32m1(0, vl);
    for ( ; i < (size_t)height; ++i) {
        // std::cout << std::endl;
        // std::cout << "i:" << i << std::endl;
        const uchar* src_row = src_data + i * src_step;
        const uchar* mask_row = mask ? (mask + i * mask_step) : nullptr;
        size_t j = 0 ;
        vl = __riscv_vsetvl_e8m1(width-j);
        for ( ; j < (size_t)width; j+=vl) {
            vl = __riscv_vsetvl_e8m1(width-j); // tail elements
            // std::cout << "j:" << j << " vl: " << vl << std::endl;

            // Load src[row][i .. i+vl]
            vuint8m1_t pixel_vector = __riscv_vle8_v_u8m1(src_row + j, vl);

            // // print pixel
            // uint8_t data[vl];
            // __riscv_vse8_v_u8m1(data, pixel_vector, vl);
            // std::cout << "pixel_vector: ";
            // for(size_t k=0; k<vl; k++) {
            //     std::cout  << (int)data[k] << " ";
            // }
            // std::cout  << std::endl;

            if(mask) {
                // Load mask[row][i .. i+vl]
                vuint8m1_t mask_load = __riscv_vle8_v_u8m1(mask_row + j, vl);
                vbool8_t mask_vector =  __riscv_vmsne_vx_u8m1_b8(mask_load, 0, vl);

                // // print mask
                // std::cout << "mask_load: ";
                // for(size_t k=0; k<vl; ++k) {
                //     std::cout << (bool)mask_row[k] << " ";
                // }
                // std::cout << std::endl;
                // uchar maskdata[vl];
                // __riscv_vsm_v_b8(maskdata, mask_vector, vl);
                // std::cout << "mask_vector: ";
                // for(size_t k=0; k<vl; ++k) {
                //     std::cout  << (bool)maskdata[k] << " ";
                // }
                // std::cout  << std::endl;

                vec_sum = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector, pixel_vector, u16_zero, vl);
                if(stddev_val) {
                    auto pixel_squared_vector = __riscv_vwmulu_vv_u16m2_m(mask_vector, pixel_vector, pixel_vector, vl);
                    vec_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector, pixel_squared_vector, u32_zero, vl);
                }
                total_count +=  __riscv_vcpop_m_b8(mask_vector, vl);
            }
            else {
                vec_sum = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector, u16_zero, vl);
                if(stddev_val) {
                    auto pixel_squared_vector = __riscv_vwmulu_vv_u16m2(pixel_vector, pixel_vector, vl);
                    vec_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector, u32_zero, vl);
                }
                total_count +=  vl;
            }
            auto temp_sum = __riscv_vmv_x_s_u16m1_u16(vec_sum);
            sum += static_cast<uint64_t>(temp_sum);
            if (stddev_val) {
                auto temp_sqsum = __riscv_vmv_x_s_u32m1_u32(vec_sqsum);
                sqsum += static_cast<uint64_t>(temp_sqsum);
            }
            std::cout << "temp_sum: " << temp_sum << " sum: " << sum << " total_count: " << total_count << std::endl;
        }
    }

    if (total_count == 0)
    {
        if (mean_val) *mean_val = 0.0;
        if (stddev_val) *stddev_val = 0.0;
        return CV_HAL_ERROR_OK;
    }
    std::cout << "sum: " << sum << "sqsum: " << sqsum << " total_count: " << total_count << std::endl;
    // Return values
    double mean = static_cast<double>(sum) / total_count;
    if (mean_val) {
        *mean_val = mean;
    }
    std::cout << "mean: " << mean << std::endl;
    if (stddev_val) {
        double variance = std::max((sqsum / total_count) - (mean * mean), 0.0);
        std::cout << "variance: " << variance << std::endl;
        double stddev = std::sqrt(variance);
        std::cout << "stddev: " << stddev << std::endl;
        *stddev_val = stddev;
    }
    return CV_HAL_ERROR_OK;
}

inline int meanStdDev_8uc4(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
    for(size_t i=0; i<4; ++i) {
        double mean = 0.0, stddev = 0.0;
        const uchar* src_c = src_data + i * height * src_step;
        if(!stddev_val)  {
            meanStdDev_8uc1(src_c, src_step, width, height, &mean, nullptr, mask, mask_step);
            mean_val[i] = mean;
        }
        else {
            meanStdDev_8uc1(src_c, src_step, width, height, &mean, &stddev, mask, mask_step);
            mean_val[i] = mean;
            stddev_val[i] = stddev;
        }
    }
    return CV_HAL_ERROR_OK;
}

inline int meanStdDev_32fc1(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
    size_t loop_unroll = 4;
    size_t loop_rem = height % loop_unroll;
    // initialize variables
    size_t total_count = 0;
    
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0.0, vlmax);
    vfloat32m1_t vec_sqsum = __riscv_vfmv_v_f_f32m1(0.0, vlmax);
    vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(0.0, vlmax);
    vfloat32m1_t temp_sqsum = __riscv_vfmv_v_f_f32m1(0.0, vlmax);
    vfloat64m1_t vec_f64_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    int j = 0;
    int vl = __riscv_vsetvl_e8m1(width-j);
    size_t i=0;
    for ( ; i < height - loop_rem; i+=4) {
        const float* src_row1 = reinterpret_cast<const float*>(src_data) + i * src_step;
        const float* src_row2 = reinterpret_cast<const float*>(src_data) + (i+1) * src_step;
        const float* src_row3 = reinterpret_cast<const float*>(src_data) + (i+2) * src_step;
        const float* src_row4 = reinterpret_cast<const float*>(src_data) + (i+3) * src_step;
        const u_int32_t* mask_row1 = mask ? reinterpret_cast<const u_int32_t*>(mask + i * mask_step) : nullptr;
        const u_int32_t* mask_row2 = mask ? reinterpret_cast<const u_int32_t*>(mask + ( i + 1 ) * mask_step) : nullptr;
        const u_int32_t* mask_row3 = mask ? reinterpret_cast<const u_int32_t*>(mask + ( i + 2 ) * mask_step) : nullptr;
        const u_int32_t* mask_row4 = mask ? reinterpret_cast<const u_int32_t*>(mask + ( i + 3 ) * mask_step) : nullptr;
        for ( ; j < width; ) {
            vl = __riscv_vsetvl_e8m1(width-j); // tail elements
            // Load src[row][i .. i+vl]
            vfloat32m1_t pixel_vector1 = __riscv_vle32_v_f32m1(src_row1 + j, vl);
            vfloat32m1_t pixel_vector2 = __riscv_vle32_v_f32m1(src_row2 + j, vl);
            vfloat32m1_t pixel_vector3 = __riscv_vle32_v_f32m1(src_row3 + j, vl);
            vfloat32m1_t pixel_vector4 = __riscv_vle32_v_f32m1(src_row4 + j, vl);
            vfloat32m1_t pixel_squared_vector1;
            vfloat32m1_t pixel_squared_vector2;
            vfloat32m1_t pixel_squared_vector3;
            vfloat32m1_t pixel_squared_vector4;
            if(mask) {
                // Load mask[row][i .. i+vl]
                vuint32m1_t mask_load1 = __riscv_vle32_v_u32m1(mask_row1 + j, vl);
                vuint32m1_t mask_load2 = __riscv_vle32_v_u32m1(mask_row2 + j, vl);
                vuint32m1_t mask_load3 = __riscv_vle32_v_u32m1(mask_row3 + j, vl);
                vuint32m1_t mask_load4 = __riscv_vle32_v_u32m1(mask_row4 + j, vl);
                vbool32_t mask_vector1 =  __riscv_vmsne_vx_u32m1_b32(mask_load1, 0, vl);
                vbool32_t mask_vector2 =  __riscv_vmsne_vx_u32m1_b32(mask_load2, 0, vl);
                vbool32_t mask_vector3 =  __riscv_vmsne_vx_u32m1_b32(mask_load3, 0, vl);
                vbool32_t mask_vector4 =  __riscv_vmsne_vx_u32m1_b32(mask_load4, 0, vl);

                // vec_s[0] <- sum(vec_s[0] , pixel_vector[*]) , if not masked
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector1, pixel_vector1, vec_s, vl);
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector2, pixel_vector2, vec_s, vl);
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector3, pixel_vector3, vec_s, vl);
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector4, pixel_vector4, vec_s, vl);
                if(stddev_val) {
                    // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*] , if not masked
                    pixel_squared_vector1 = __riscv_vfmul_vv_f32m1_m(mask_vector1, pixel_vector1, pixel_vector1, vl);
                    pixel_squared_vector2 = __riscv_vfmul_vv_f32m1_m(mask_vector2, pixel_vector2, pixel_vector2, vl);
                    pixel_squared_vector3 = __riscv_vfmul_vv_f32m1_m(mask_vector3, pixel_vector3, pixel_vector3, vl);
                    pixel_squared_vector4 = __riscv_vfmul_vv_f32m1_m(mask_vector4, pixel_vector4, pixel_vector4, vl);
                    // sqsum[0] <- sum(temp_sqsum[0] , pixel_squared_vector[*]) , if not masked
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector1, pixel_squared_vector1, temp_sqsum, vl);
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector2, pixel_squared_vector2, temp_sqsum, vl);
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector3, pixel_squared_vector3, temp_sqsum, vl);
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector4, pixel_squared_vector4, temp_sqsum, vl);
                }
            }
            else {
                // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*])
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1(pixel_vector1, vec_s, vl);
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1(pixel_vector2, vec_s, vl);
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1(pixel_vector3, vec_s, vl);
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1(pixel_vector4, vec_s, vl);
                if(stddev_val) {
                    // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*]
                    pixel_squared_vector1 = __riscv_vfmul_vv_f32m1(pixel_vector1, pixel_vector1, vl);
                    pixel_squared_vector2 = __riscv_vfmul_vv_f32m1(pixel_vector2, pixel_vector2, vl);
                    pixel_squared_vector3 = __riscv_vfmul_vv_f32m1(pixel_vector3, pixel_vector3, vl);
                    pixel_squared_vector4 = __riscv_vfmul_vv_f32m1(pixel_vector4, pixel_vector4, vl);
                    // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*])
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1(pixel_squared_vector1, temp_sqsum, vl);
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1(pixel_squared_vector2, temp_sqsum, vl);
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1(pixel_squared_vector3, temp_sqsum, vl);
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1(pixel_squared_vector4, temp_sqsum, vl);
                }
            }
            // vec_sum[0] = sum( temp_sum , vec_sum)
            vec_sum = __riscv_vfadd_vv_f32m1(vec_s, vec_sum, vl);
            if (stddev_val) {
                vec_sqsum = __riscv_vfadd_vv_f32m1(temp_sqsum, vec_sqsum, vl);
            }
            total_count += vl;
            j += vl;
        }
    }
    for ( ; i < height; ++i) {
        const float* src_row = reinterpret_cast<const float*>(src_data) + i * src_step;
        const u_int32_t* mask_row = mask ? reinterpret_cast<const u_int32_t*>(mask + i * mask_step) : nullptr;
        for ( ; j < width; ) {
            vl = __riscv_vsetvl_e32m1(width-j); // tail elements
            // Load src[row][i .. i+vl]
            vfloat32m1_t pixel_vector = __riscv_vle32_v_f32m1(src_row + j, vl);
            vfloat32m1_t pixel_squared_vector;
            if(mask) {
                // Load mask[row][i .. i+vl]
                vuint32m1_t mask_load = __riscv_vle32_v_u32m1(mask_row + j, vl);
                vbool32_t mask_vector =  __riscv_vmsne_vx_u32m1_b32(mask_load, 0, vl);
                // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*]) , if not masked
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector, pixel_vector, vec_s, vl);
                if(stddev_val) {
                    // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*] , if not masked
                    pixel_squared_vector = __riscv_vfmul_vv_f32m1_m(mask_vector, pixel_vector, pixel_vector, vl);
                    // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*]) , if not masked
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector, pixel_squared_vector, temp_sqsum, vl);
                }
            }
            else {
                // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*])
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1(pixel_vector, vec_s, vl);
                if(stddev_val) {
                    // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*]
                    pixel_squared_vector = __riscv_vfmul_vv_f32m1(pixel_vector, pixel_vector, vl);
                    // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*])
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1(pixel_squared_vector, temp_sqsum, vl);
                }
            }
            // vec_sum[0] = sum( temp_sum , vec_sum)
            vec_sum = __riscv_vfadd_vv_f32m1(vec_s, vec_sum, vl);
            if (stddev_val) {
                vec_sqsum = __riscv_vfadd_vv_f32m1(temp_sqsum, vec_sqsum, vl);
            }
            total_count += vl;
            j += vl;
        }
    }
    if (total_count == 0)
    {
        if (mean_val) *mean_val = 0.0;
        if (stddev_val) *stddev_val = 0.0;
        return CV_HAL_ERROR_OK;
    }
    // Return values
    vfloat64m1_t float_sum = __riscv_vfwredusum_vs_f32m1_f64m1(vec_sum, vec_f64_zero, vl);
    double sum = __riscv_vfmv_f_s_f64m1_f64(float_sum);
    double mean = sum / total_count;
    if (mean_val) {
        *mean_val = mean;
    }
    if (stddev_val) {
        vfloat64m1_t float_sqsum = __riscv_vfwredusum_vs_f32m1_f64m1(vec_sum, vec_f64_zero, vl);
        double sqsum = __riscv_vfmv_f_s_f64m1_f64(float_sqsum);
        double variance = std::max((sqsum / total_count) - (mean * mean), 0.0);
        double stddev = std::sqrt(variance);
        *stddev_val = stddev;
    }
    return CV_HAL_ERROR_OK;
}
}}
#endif


