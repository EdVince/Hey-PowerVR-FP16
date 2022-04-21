#include <stdio.h>
#include <mat.h>
#include <iostream>
#include <arm_neon.h>

using namespace std;
using namespace ncnn;


void test_arm_fp16a(float a_fp32, float b_fp32)
{
    unsigned short a_fp16 = float32_to_float16(a_fp32);
    unsigned short a_fp16x8[8] = {a_fp16,a_fp16,a_fp16,a_fp16,a_fp16,a_fp16,a_fp16,a_fp16};
    float a_fp32rec = float16_to_float32(a_fp16);

    unsigned short b_fp16 = float32_to_float16(b_fp32);
    unsigned short b_fp16x8[8] = {b_fp16,b_fp16,b_fp16,b_fp16,b_fp16,b_fp16,b_fp16,b_fp16};
    float b_fp32rec = float16_to_float32(b_fp16);

    float c_fp32;
    unsigned short c_fp16;
    unsigned short c_fp16x8[8] = {0};


    const __fp16* ptr = (__fp16*)a_fp16x8;
    const __fp16* ptr1 = (__fp16*)b_fp16x8;
    __fp16* outptr = (__fp16*)c_fp16x8;
    float16x8_t _p = vld1q_f16(ptr);
    float16x8_t _p1 = vld1q_f16(ptr1);
    float16x8_t _outp = vmulq_f16(_p, _p1);
    vst1q_f16(outptr, _outp);


    c_fp16 = c_fp16x8[0];
    c_fp32 = float16_to_float32(c_fp16);

    printf("---- testing ARM fp16a ----\n");
    printf("fp32->fp32: %.17f * %.17f = %.17f\n",a_fp32,b_fp32,a_fp32*b_fp32);
    printf("fp16->fp16: %.17f * %.17f = %.17f(%.6lf%%)\n",a_fp32rec,b_fp32rec,c_fp32,100.0*std::abs(double(a_fp32rec)*double(b_fp32rec)-double(c_fp32))/(double(a_fp32rec)*double(b_fp32rec)));
}

int main()
{
    test_arm_fp16a(0.33f,0.2f);

    return 0;
}
