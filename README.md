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
```
**[荣耀畅玩30Plus](https://benchmarks.ul.com/hardware/phone/Huawei+Honor+Play+30+Plus+review)**：
```
CPU: Dimensity 700 (2.2 GHz 2 * ARM Cortex-A76 & 2.0 GHz 6 * ARM Cortex-A55)
[0 Mali-G57 MC2]  queueC=0[2]  queueG=0[2]  queueT=0[2]
[0 Mali-G57 MC2]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[0 Mali-G57 MC2]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 Mali-G57 MC2]  subgroup=16  basic=1  vote=1  ballot=1  shuffle=0
```
**[三星A12](https://benchmarks.ul.com/hardware/phone/Samsung+Galaxy+A12+review)**：
```
CPU: Helio P35 (2.3 GHz 4 * ARM Cortex-A53 & 1.8 GHz 4 * ARM Cortex-A53)
GPU: GE8320 (maybe 8 for Series8XE/Plus series, 3 for 6 pixels/clock, 2 for 128 FP16 FLOPs/clock and 0 for no PVRIC)
[0 PowerVR Rogue GE8320]  queueC=0[2]  queueG=0[2]  queueT=0[2]
[0 PowerVR Rogue GE8320]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[0 PowerVR Rogue GE8320]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 PowerVR Rogue GE8320]  subgroup=1  basic=1  vote=1  ballot=1  shuffle=1
```

## ncnn分析

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


## 参考
1. [ncnn](https://github.com/Tencent/ncnn)
2. [tnn](https://github.com/Tencent/TNN)
3. [mnn](https://github.com/alibaba/MNN)