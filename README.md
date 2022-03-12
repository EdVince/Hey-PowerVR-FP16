### Fucking-PowerVR-FP16
最近工作的时候遇到了一个很神奇的bug，PowerVR的GPU：

1. 在ncnn的vulkan后端上，使用use_fp16_arithmetic=true会有较大的计算误差
2. 在tnn的opencl后端上，使用PRECISION_AUTO同样会有较大的计算误差

因此这里开个repo专门分析一个这个问题。

### 工作步骤
- [ ] 用pytorch->onnx捏一个只含有一个卷积层的测试模型
- [ ] ncnn+vulkan+app的PowerVR、Mali、Adreno的FP32+FP16
- [ ] tnn+opencl+app的PowerVR、Mali、Adreno的FP32+FP16
- [ ] mnn+opencl+app的PowerVR、Mali、Adreno的FP32+FP16
- [ ] ncnn+vulkan+shell的PowerVR、Mali、Adreno的FP32+FP16
- [ ] tnn+opencl+shell的PowerVR、Mali、Adreno的FP32+FP16
- [ ] mnn+opencl+shell的PowerVR、Mali、Adreno的FP32+FP16
- [ ] vulkan+shell的PowerVR、Mali、Adreno的FP32+FP16
- [ ] opencl+shell的PowerVR、Mali、Adreno的FP32+FP16

### 工作分析

### 参考
1. [ncnn](https://github.com/Tencent/ncnn)
2. [tnn](https://github.com/Tencent/TNN)
3. [mnn](https://github.com/alibaba/MNN)