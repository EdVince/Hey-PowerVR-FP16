# Fucking-PowerVR-FP16
最近工作的时候遇到了一个很神奇的bug，PowerVR的GPU：

1. 在ncnn的vulkan后端上，使用use_fp16_arithmetic=true会有较大的计算误差
2. 在tnn的opencl后端上，使用PRECISION_AUTO同样会有较大的计算误差

因此这里开个repo专门分析一个这个问题。

## 测试设备

**[三星S9](https://benchmarks.ul.com/hardware/phone/Samsung+Galaxy+S9+%28SDM845%29+review)**：
```
CPU: Snapdragon 845 (2.8 GHz 4 * Kyro 385 & 1.7 GHz 4 * Kyro 385)
[0 Adreno (TM) 630]  queueC=0[3]  queueG=0[3]  queueT=0[3]
[0 Adreno (TM) 630]  bugsbn1=1  bugbilz=0  bugcopc=0  bugihfa=1
[0 Adreno (TM) 630]  fp16-p/s/a=1/0/1  int8-p/s/a=1/0/0
[0 Adreno (TM) 630]  subgroup=64  basic=1  vote=1  ballot=0  shuffle=0

For OpenCL
---- 0 Platform Name: QUALCOMM Snapdragon(TM)
---- 0 Platform Vendor: QUALCOMM
---- 0 Platform Version: OpenCL 2.0 QUALCOMM build: commit #fdd61e0 changeid #I20154638fb Date: 10/07/20 Wed Local Branch:  Remote Branch: refs/tags/AU_LINUX_ANDROID_LA.UM.8.3.R1.10.00.00.520.058
---- ---- 0 Device Name: QUALCOMM Adreno(TM)
---- ---- 0 Device Vendor: QUALCOMM
---- ---- 0 Device Version: OpenCL 2.0 Adreno(TM) 630
---- ---- 0 Driver Version: OpenCL 2.0 QUALCOMM build: commit #fdd61e0 changeid #I20154638fb Date: 10/07/20 Wed Local Branch:  Remote Branch: refs/tags/AU_LINUX_ANDROID_LA.UM.8.3.R1.10.00.00.520.058 Compiler E031.37.03.00
---- ---- 0 Device OpenCL C Version: OpenCL C 2.0 Adreno(TM) 630
---- ---- 0 Device Type: GPU
---- ---- ---- 0 Device half Preferred/Native Vector Sizes: 1 / 1
---- ---- ---- 0 Device half Denormals:                       no
---- ---- ---- 0 Device half Infinity and NANs:               yes
---- ---- ---- 0 Device half Round to nearest:                yes
---- ---- ---- 0 Device half Round to zero:                   no
---- ---- ---- 0 Device half Round to infinity:               yes
---- ---- ---- 0 Device half IEEE754-2008 fused multiply-add: no
---- ---- ---- 0 Device half Support is emulated in software: no

```
**[荣耀畅玩30Plus](https://benchmarks.ul.com/hardware/phone/Huawei+Honor+Play+30+Plus+review)**：
```
CPU: Dimensity 700 (2.2 GHz 2 * ARM Cortex-A76 & 2.0 GHz 6 * ARM Cortex-A55)
[0 Mali-G57 MC2]  queueC=0[2]  queueG=0[2]  queueT=0[2]
[0 Mali-G57 MC2]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[0 Mali-G57 MC2]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 Mali-G57 MC2]  subgroup=16  basic=1  vote=1  ballot=1  shuffle=0

For OpenCL
---- 0 Platform Name: ARM Platform
---- 0 Platform Vendor: ARM
---- 0 Platform Version: OpenCL 2.1 v1.r26p0-01eac0.478dcdf9b4e0cb2898e119ab91ed3dcf
---- ---- 0 Device Name: Mali-G57 MC2 r0p1
---- ---- 0 Device Vendor: ARM
---- ---- 0 Device Version: OpenCL 2.1 v1.r26p0-01eac0.478dcdf9b4e0cb2898e119ab91ed3dcf
---- ---- 0 Driver Version: 2.1
---- ---- 0 Device OpenCL C Version: OpenCL C 2.0 v1.r26p0-01eac0.478dcdf9b4e0cb2898e119ab91ed3dcf
---- ---- 0 Device Type: GPU
---- ---- ---- 0 Device half Preferred/Native Vector Sizes: 8 / 2
---- ---- ---- 0 Device half Denormals:                       yes
---- ---- ---- 0 Device half Infinity and NANs:               yes
---- ---- ---- 0 Device half Round to nearest:                yes
---- ---- ---- 0 Device half Round to zero:                   yes
---- ---- ---- 0 Device half Round to infinity:               yes
---- ---- ---- 0 Device half IEEE754-2008 fused multiply-add: yes
---- ---- ---- 0 Device half Support is emulated in software: no
```
**[三星A12](https://benchmarks.ul.com/hardware/phone/Samsung+Galaxy+A12+review)**：
```
CPU: Helio P35 (2.3 GHz 4 * ARM Cortex-A53 & 1.8 GHz 4 * ARM Cortex-A53)
[0 PowerVR Rogue GE8320]  queueC=0[2]  queueG=0[2]  queueT=0[2]
[0 PowerVR Rogue GE8320]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[0 PowerVR Rogue GE8320]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 PowerVR Rogue GE8320]  subgroup=1  basic=1  vote=1  ballot=1  shuffle=1

For OpenCL
---- 0 Platform Name: PowerVR
---- 0 Platform Vendor: Imagination Technologies
---- 0 Platform Version: OpenCL 1.2 
---- ---- 0 Device Name: PowerVR GE8320
---- ---- 0 Device Vendor: Imagination Technologies
---- ---- 0 Device Version: OpenCL 1.2 
---- ---- 0 Driver Version: 1.13@5776728
---- ---- 0 Device OpenCL C Version: OpenCL C 1.2 
---- ---- 0 Device Type: GPU
---- ---- ---- 0 Device half Preferred/Native Vector Sizes: 0 / 0
---- ---- ---- 0 Device half Denormals:                       no
---- ---- ---- 0 Device half Infinity and NANs:               yes
---- ---- ---- 0 Device half Round to nearest:                no
---- ---- ---- 0 Device half Round to zero:                   yes
---- ---- ---- 0 Device half Round to infinity:               no
---- ---- ---- 0 Device half IEEE754-2008 fused multiply-add: yes
---- ---- ---- 0 Device half Support is emulated in software: no
```

## 简单的ncnn测试

##### 源码编译：
1. 源码版本20220216
2. 在testutil.h中注释掉两个"FIXME fp16a may produce large error"使其编译出vk+fp16a的代码
3. 命令：
```
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-24 -DNCNN_VULKAN=ON -DNCNN_BUILD_TESTS=ON ..
make -j$(nproc)
```
4. 只编译sigmoid和softmax算子进行测试

##### 结果：

| fp16a   | Naive    | Adreno   | Mali     | PowerVR  |
| ------- | -------- | -------- | -------- | -------- |
| Softmax | 0.013287 | 0.013306 | 0.013329 | 0.013542 |
| Sigmoid | 0.684264 | 0.684082 | 0.684570 | 0.684082 |
| ReLU    | -0.078906 | -0.078857 | -0.078857 | -0.078857 |

## OpenCL测试

##### 代码编译
1. OPenCL头文件从这里获取: https://github.com/KhronosGroup/OpenCL-Headers
2. armeabi-v7a平台编译用的libOpenCL.so从这里获取: https://github.com/Tencent/TNN/blob/master/third_party/opencl/lib/libOpenCL.so

##### 结果：

Formula: a * b = c

| Device  | a(fp16)              | b(fp16)              | c(fp16)              |
| ------- | -------------------- | -------------------- | -------------------- |
| UHD630  | 0.19995117187500000f | 0.32983398437500000f | 0.06597900390625000f |
| Adreno  | 0.19995117187500000f | 0.32983398437500000f | 0.06597900390625000f |
| Mali    | 0.19995117187500000f | 0.32983398437500000f | 0.06597900390625000f |
| PowerVR | 0.19995117187500000f | 0.32983398437500000f | 0.06591796875000000f |

问题: ncnn的vulkan结果显示powervr是支持fp16p/s/a的，但opencl返回CL_DEVICE_PREFERRED_VECTOR_WIDTH_
HALF==0意味着它不支持cl_khr_fp16扩展

## 参考
1. [ncnn arm](https://github.com/Tencent/ncnn/tree/master/src/layer/arm)
2. [ncnn vk](https://github.com/Tencent/ncnn/tree/master/src/layer/vulkan)
3. [arm arm&cl](https://github.com/ARM-software/ComputeLibrary)
4. [google vk](https://github.com/google/uVkCompute)
5. [IEEE 2019](https://ieeexplore.ieee.org/document/8766229)
6. [IEEE 2019 wiki](https://en.wikipedia.org/wiki/IEEE_754#2019)
7. [IEEE 754 Converter](https://www.h-schmidt.net/FloatConverter/IEEE754.html)
8. [XiangShan fudian](https://github.com/OpenXiangShan/fudian)
9. [opencl info](https://github.com/Oblomov/clinfo)
10. [opencl 1.2](https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf)