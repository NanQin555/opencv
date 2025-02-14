// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include <riscv_vector.h>
#include <iostream>
#include <cassert>
namespace cv { namespace cv_hal_rvv {

// RVV_TYP:
// float16, float32, float64, (u)int8, (u)int16, (u)int32, (u)int64
// pre: 1(sign/unsign) 1(float/int) 00 (8, 16, 32, 64)
// mf8, mf4, mf2, m1, m2, m4, m8
// suf: 0000 (for 7 type)

#define RVV_FLOAT16 0b1101
#define RVV_FLOAT32 0b1110
#define RVV_FLOAT64 0b1111
#define RVV_INT8    0b1000
#define RVV_INT16   0b1001
#define RVV_INT32   0b1010
#define RVV_INT64   0b1011
#define RVV_UINT8   0b0000
#define RVV_UINT16  0b0001
#define RVV_UINT32  0b0010
#define RVV_UINT64  0b0011

#define RVV_MF8 0b0000
#define RVV_MF4 0b0001
#define RVV_MF2 0b0010
#define RVV_M1  0b0011
#define RVV_M2  0b0100
#define RVV_M4  0b0101
#define RVV_M8  0b0110

#define RVV_TYPE_MASK 0b11110000
#define RVV_LMUL_MASK 0b00001111

#define RVV_COMBINE(TYPE, LMUL)  (((TYPE) << 4) | (LMUL))
#define RVV_TYPE(TYPE) ((TYPE & RVV_TYPE_MASK) >> 4)
#define RVV_LMUL(TYPE) (TYPE & RVV_LMUL_MASK)

#define VFLOAT16MF4 RVV_COMBINE(RVV_FLOAT16, RVV_MF4)
#define VFLOAT16MF2 RVV_COMBINE(RVV_FLOAT16, RVV_MF2)
#define VFLOAT16M1 RVV_COMBINE(RVV_FLOAT16, RVV_M1)
#define VFLOAT16M2 RVV_COMBINE(RVV_FLOAT16, RVV_M2)
#define VFLOAT16M4 RVV_COMBINE(RVV_FLOAT16, RVV_M4)
#define VFLOAT16M8 RVV_COMBINE(RVV_FLOAT16, RVV_M8)
#define VFLOAT32MF2 RVV_COMBINE(RVV_FLOAT32, RVV_MF2)
#define VFLOAT32M1 RVV_COMBINE(RVV_FLOAT32, RVV_M1)
#define VFLOAT32M2 RVV_COMBINE(RVV_FLOAT32, RVV_M2)
#define VFLOAT32M4 RVV_COMBINE(RVV_FLOAT32, RVV_M4)
#define VFLOAT32M8 RVV_COMBINE(RVV_FLOAT32, RVV_M8)
#define VFLOAT64M1 RVV_COMBINE(RVV_FLOAT64, RVV_M1)
#define VFLOAT64M2 RVV_COMBINE(RVV_FLOAT64, RVV_M2)
#define VFLOAT64M4 RVV_COMBINE(RVV_FLOAT64, RVV_M4)
#define VFLOAT64M8 RVV_COMBINE(RVV_FLOAT64, RVV_M8)
#define VINT8MF8 RVV_COMBINE(RVV_INT8, RVV_MF8)
#define VINT8MF4 RVV_COMBINE(RVV_INT8, RVV_MF4)
#define VINT8MF2 RVV_COMBINE(RVV_INT8, RVV_MF2)
#define VINT8M1 RVV_COMBINE(RVV_INT8, RVV_M1)
#define VINT8M2 RVV_COMBINE(RVV_INT8, RVV_M2)
#define VINT8M4 RVV_COMBINE(RVV_INT8, RVV_M4)
#define VINT8M8 RVV_COMBINE(RVV_INT8, RVV_M8)
#define VINT16MF4 RVV_COMBINE(RVV_INT16, RVV_MF4)
#define VINT16MF2 RVV_COMBINE(RVV_INT16, RVV_MF2)
#define VINT16M1 RVV_COMBINE(RVV_INT16, RVV_M1)
#define VINT16M2 RVV_COMBINE(RVV_INT16, RVV_M2)
#define VINT16M4 RVV_COMBINE(RVV_INT16, RVV_M4)
#define VINT16M8 RVV_COMBINE(RVV_INT16, RVV_M8)
#define VINT32MF2 RVV_COMBINE(RVV_INT32, RVV_MF2)
#define VINT32M1 RVV_COMBINE(RVV_INT32, RVV_M1)
#define VINT32M2 RVV_COMBINE(RVV_INT32, RVV_M2)
#define VINT32M4 RVV_COMBINE(RVV_INT32, RVV_M4)
#define VINT32M8 RVV_COMBINE(RVV_INT32, RVV_M8)
#define VINT64M1 RVV_COMBINE(RVV_INT64, RVV_M1)
#define VINT64M2 RVV_COMBINE(RVV_INT64, RVV_M2)
#define VINT64M4 RVV_COMBINE(RVV_INT64, RVV_M4)
#define VINT64M8 RVV_COMBINE(RVV_INT64, RVV_M8)
#define VUINT8MF8 RVV_COMBINE(RVV_INT8, RVV_MF8)
#define VUINT8MF4 RVV_COMBINE(RVV_INT8, RVV_MF4)
#define VUINT8MF2 RVV_COMBINE(RVV_INT8, RVV_MF2)
#define VUINT8M1 RVV_COMBINE(RVV_INT8, RVV_M1)
#define VUINT8M2 RVV_COMBINE(RVV_INT8, RVV_M2)
#define VUINT8M4 RVV_COMBINE(RVV_INT8, RVV_M4)
#define VUINT8M8 RVV_COMBINE(RVV_INT8, RVV_M8)
#define VUINT16MF4 RVV_COMBINE(RVV_INT16, RVV_MF4)
#define VUINT16MF2 RVV_COMBINE(RVV_INT16, RVV_MF2)
#define VUINT16M1 RVV_COMBINE(RVV_INT16, RVV_M1)
#define VUINT16M2 RVV_COMBINE(RVV_INT16, RVV_M2)
#define VUINT16M4 RVV_COMBINE(RVV_INT16, RVV_M4)
#define VUINT16M8 RVV_COMBINE(RVV_INT16, RVV_M8)
#define VUINT32MF2 RVV_COMBINE(RVV_INT32, RVV_MF2)
#define VUINT32M1 RVV_COMBINE(RVV_INT32, RVV_M1)
#define VUINT32M2 RVV_COMBINE(RVV_INT32, RVV_M2)
#define VUINT32M4 RVV_COMBINE(RVV_INT32, RVV_M4)
#define VUINT32M8 RVV_COMBINE(RVV_INT32, RVV_M8)
#define VUINT64M1 RVV_COMBINE(RVV_INT64, RVV_M1)
#define VUINT64M2 RVV_COMBINE(RVV_INT64, RVV_M2)
#define VUINT64M4 RVV_COMBINE(RVV_INT64, RVV_M4)
#define VUINT64M8 RVV_COMBINE(RVV_INT64, RVV_M8)

static inline std::string suffix(int type) {
    string suffix;
    switch (RVV_TYPE(type)) {
        case RVV_FLOAT16:
            suffix += "f16";
            break;
        case RVV_FLOAT32;
            suffix += "f32";
            break;
        case RVV_FLOAT64;
            suffix += "f64";
            break;
        case RVV_INT8;
            suffix += "i8";
            break;
        case RVV_INT16;
            suffix += "i16";
            break;
        case RVV_INT32;
            suffix += "i32";
            break;
        case RVV_INT64;
            suffix += "i64";
            break;
        case RVV_UINT8;
            suffix += "u8";
            break;
        case RVV_UINT16;
            suffix += "u16";
            break;
        case RVV_UINT32;
            suffix += "u32";
            break;
        case RVV_UINT64;
            suffix += "u64";
            break;
        default
            assert(0 && "Invalid rvv type!");
            break;
    }
    switch (RVV_LMUL(type)) {
        case RVV_MF8:
            suffix += "mf8";
            break;
        case RVV_MF4:
            suffix += "mf4";
            break;
        case RVV_MF2:
            suffix += "mf2";
            break;
        case RVV_M1:
            suffix += "m1";
            break;
        case RVV_M2:
            suffix += "m2";
            break;
        case RVV_M4:
            suffix += "m4";
            break;
        case RVV_M8:
            suffix += "m8";
            break;
        default:
            assert(0 && "Invalid rvv type!");
            break;
    }
    return suffix;
}



/**
 * 
 */

// Explicit (Non-overloaded) intrinsics
// vfloat16mf4_t __riscv_vle16_v_f16mf4(const _Float16 *rs1, size_t vl);

// Explicit (Non-overloaded) intrinsics, policy variants
// vfloat16mf4_t __riscv_vle16_v_f16mf4_tu(vfloat16mf4_t vd, const _Float16 *rs1,size_t vl);

// Implicit (Overloaded) intrinsics
// vfloat16mf4_t __riscv_vle16(vbool64_t vm, const _Float16 *rs1, size_t vl);

// Implicit (Overloaded) intrinsics, policy variants
// vfloat16mf4_t __riscv_vle16_tu(vfloat16mf4_t vd, const _Float16 *rs1, size_t vl);

// __riscv_v##intrinsic##ret_type

// different type of source -> ret type  


} // cv_hal_rvv::
} // cv::
