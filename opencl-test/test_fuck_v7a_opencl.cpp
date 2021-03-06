#include <stdio.h>
#include <mat.h>
#include <iostream>
#include <sstream>
#include "CL/opencl.h"
#include <stdlib.h>
using namespace std;
using namespace ncnn;

string getPlatformInfo(const cl_platform_id pid, cl_uint type) {
	size_t param_value_size;
	clGetPlatformInfo(pid, type, 0, NULL, &param_value_size);
	char* param_value = new char[param_value_size];
	clGetPlatformInfo(pid, type, param_value_size, param_value, NULL);
	return param_value;
}

string getDeviceInfo(const cl_device_id pid, cl_uint type) {
	if (type == CL_DEVICE_TYPE)
	{
		unsigned long long dt;
		clGetDeviceInfo(pid, type, 8, &dt, NULL);
		if(dt == CL_DEVICE_TYPE_DEFAULT)
			return "DEFAULT";
		else if (dt == CL_DEVICE_TYPE_CPU)
			return "CPU";
		else if (dt == CL_DEVICE_TYPE_GPU)
			return "GPU";
		else if (dt == CL_DEVICE_TYPE_ACCELERATOR)
			return "ACCELERATOR";
		else if (dt == CL_DEVICE_TYPE_CUSTOM)
			return "CUSTOM";
		else if (dt == CL_DEVICE_TYPE_ALL)
			return "ALL";
		else
			return "FUCK";
	}
	else if (type == CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF || type == CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF) 
	{
		unsigned long dt;
		clGetDeviceInfo(pid, type, 4, &dt, NULL);
		return std::to_string(dt);
	}
	else if (type == 0x1033 || type == CL_DEVICE_SINGLE_FP_CONFIG) {
		unsigned long long dt;
		clGetDeviceInfo(pid, type, 8, &dt, NULL);
		return std::to_string(dt);
	}
	size_t param_value_size;
	clGetDeviceInfo(pid, type, 0, NULL, &param_value_size);
	char* param_value = new char[param_value_size];
	clGetDeviceInfo(pid, type, param_value_size, param_value, NULL);
	return param_value;
}

size_t shrRoundUp(size_t localWorkSize, size_t numItems) {
	size_t result = localWorkSize;
	while (result < numItems)
		result += localWorkSize;
	return result;
}


void vector_mul_cpu_fp32(const float* const src_a, const float* const src_b, float* const res, const int size)
{
	for (int i = 0; i < size; i++) {
		res[i] = src_a[i] * src_b[i];
	}
}

void vector_mul_cl_fp16(cl_device_id device, const float* const src_a, const float* const src_b, float* const res, const int size)
{
	cl_int error = 0;
	cl_context context;
	cl_command_queue queue;

	//context
	context = clCreateContext(0, 1, &device, NULL, NULL, &error);
	if (error != CL_SUCCESS) {
		cout << "Error creating context: " << error << endl;
		exit(error);
	}

	// Command-queue
	queue = clCreateCommandQueue(context, device, 0, &error);
	if (error != CL_SUCCESS) {
		cout << "Error creating command queue: " << error << endl;
		exit(error);
	}

	///memory
	unsigned short* src_a_h = new unsigned short[size];
	unsigned short* src_b_h = new unsigned short[size];
	unsigned short* res_h = new unsigned short[size];
	// init vectors
	for (int i = 0; i < size; i++)
	{
		src_a_h[i] = float32_to_float16(src_a[i]);
		src_b_h[i] = float32_to_float16(src_b[i]);
	}
	//allocate device buffer
	const int mem_size = sizeof(unsigned short) * size;
	cl_mem src_a_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, (void*)src_a_h, &error);
	cl_mem src_b_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, (void*)src_b_h, &error);
	cl_mem res_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, mem_size, NULL, &error);

	// create the program
	const char* programSource =
		"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
		"__kernel void vector_add_gpu(__global const half* src_a, __global const half* src_b, __global half* res, const int num)\n"
		"{\n"
		"	const int idx = get_global_id(0); \n"
		"	half in0 = vload_half(idx,src_a); \n"
		"	half in1 = vload_half(idx,src_b); \n"
		"	half out = in0 * in1; \n"
		"	vstore_half((float)out,idx,res);"
		"}\n";
	cl_program program = clCreateProgramWithSource(context, 1, reinterpret_cast<const char**>(&programSource), NULL, &error);
	if (error != CL_SUCCESS) {
		cout << "Error creating program: " << error << endl;
		exit(error);
	}

	//builds the program
	error = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
	if (error != CL_SUCCESS) {
		cout << "Error Build program: " << error << endl;
		// shows the log
		char* build_log;
		size_t log_size;
		// 1st get the log_size
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		build_log = new char[log_size + 1];
		cout << log_size << " log_size" << endl;
		//2nd get log
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
		build_log[log_size] = '\0';
		cout << build_log << " build log" << endl;
		delete[] build_log;
		exit(error);
	}

	// check the kernel code
	size_t bufSize = strlen(programSource) + 1;
	char* programBuffer = (char*)malloc(bufSize);
	size_t program_size_ret;
	error = clGetProgramInfo(program, CL_PROGRAM_SOURCE, bufSize, programBuffer, &program_size_ret);
	if (error != CL_SUCCESS)
	{
		cout << "Error clGetProgramInfo:" << error << endl;
	}

	//extacting the kernel
	cl_kernel vector_add_k = clCreateKernel(program, "vector_add_gpu", &error);
	if (error != CL_SUCCESS) {
		cout << "Error extacting the kernel: " << error << endl;
		exit(error);
	}
	// Enqueuing parameters
	error = clSetKernelArg(vector_add_k, 0, sizeof(cl_mem), &src_a_d);
	if (error != CL_SUCCESS)
	{
		cout << "Error Enqueuing 0 parameters:" << error << endl;
	}
	error |= clSetKernelArg(vector_add_k, 1, sizeof(cl_mem), &src_b_d);
	if (error != CL_SUCCESS)
	{
		cout << "Error Enqueuing 1 parameters:" << error << endl;
	}
	error |= clSetKernelArg(vector_add_k, 2, sizeof(cl_mem), &res_d);
	if (error != CL_SUCCESS)
	{
		cout << "Error Enqueuing 2 parameters:" << error << endl;
	}
	error |= clSetKernelArg(vector_add_k, 3, sizeof(int), &size);
	if (error != CL_SUCCESS)
	{
		cout << "Error Enqueuing 3 parameters:" << error << endl;
	}

	//launching the kernel
	const size_t local_ws = 2;  // Number of work-items per work-group
	const size_t global_ws = shrRoundUp(local_ws, size);
	error = clEnqueueNDRangeKernel(queue, vector_add_k, 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
	if (error != CL_SUCCESS)
	{
		cout << "Error Enqueuing EnqueueNDRangeKernel:" << error << endl;
	}

	// check result
	unsigned short* check = new unsigned short[size];
	clEnqueueReadBuffer(queue, res_d, CL_TRUE, 0, mem_size, check, 0, NULL, NULL);
	for (int i = 0; i < size; i++)
	{
		res[i] = float16_to_float32(check[i]);
	}

	FILE* fp = fopen("log.txt", "w");
	for (int j = 0; j < size; j++) 
	{
		{
			unsigned short tmp = src_a_h[j];
			int bit_analsis[16] = { 0 };
			for (int i = 0; i < 16; i++) {
				bit_analsis[i] = tmp & 0x1;
				tmp >>= 1;
			}
			fprintf(fp, "%d ", bit_analsis[15]);
			fprintf(fp, "%d%d%d%d%d ", bit_analsis[14], bit_analsis[13], bit_analsis[12], bit_analsis[11], bit_analsis[10]);
			fprintf(fp, "%d%d%d%d%d%d%d%d%d%d:", bit_analsis[9], bit_analsis[8], bit_analsis[7], bit_analsis[6], bit_analsis[5], bit_analsis[4], bit_analsis[3], bit_analsis[2], bit_analsis[1], bit_analsis[0]);
		}
		{
			unsigned short tmp = src_b_h[j];
			int bit_analsis[16] = { 0 };
			for (int i = 0; i < 16; i++) {
				bit_analsis[i] = tmp & 0x1;
				tmp >>= 1;
			}
			fprintf(fp, "%d ", bit_analsis[15]);
			fprintf(fp, "%d%d%d%d%d ", bit_analsis[14], bit_analsis[13], bit_analsis[12], bit_analsis[11], bit_analsis[10]);
			fprintf(fp, "%d%d%d%d%d%d%d%d%d%d:", bit_analsis[9], bit_analsis[8], bit_analsis[7], bit_analsis[6], bit_analsis[5], bit_analsis[4], bit_analsis[3], bit_analsis[2], bit_analsis[1], bit_analsis[0]);
		}
		{
			unsigned short tmp = check[j];
			int bit_analsis[16] = { 0 };
			for (int i = 0; i < 16; i++) {
				bit_analsis[i] = tmp & 0x1;
				tmp >>= 1;
			}
			fprintf(fp, "%d ", bit_analsis[15]);
			fprintf(fp, "%d%d%d%d%d ", bit_analsis[14], bit_analsis[13], bit_analsis[12], bit_analsis[11], bit_analsis[10]);
			fprintf(fp, "%d%d%d%d%d%d%d%d%d%d\n", bit_analsis[9], bit_analsis[8], bit_analsis[7], bit_analsis[6], bit_analsis[5], bit_analsis[4], bit_analsis[3], bit_analsis[2], bit_analsis[1], bit_analsis[0]);
		}
	}
	fclose(fp);


	// Cleaning up
	delete[] src_a_h;
	delete[] src_b_h;
	delete[] res_h;
	delete[] check;
	clReleaseKernel(vector_add_k);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseMemObject(src_a_d);
	clReleaseMemObject(src_b_d);
	clReleaseMemObject(res_d);
}

void vector_mul_cl_fp32(cl_device_id device, const float* const src_a, const float* const src_b, float* const res, const int size)
{
	cl_int error = 0;
	cl_context context;
	cl_command_queue queue;

	//context
	context = clCreateContext(0, 1, &device, NULL, NULL, &error);
	if (error != CL_SUCCESS) {
		cout << "Error creating context: " << error << endl;
		exit(error);
	}

	// Command-queue
	queue = clCreateCommandQueue(context, device, 0, &error);
	if (error != CL_SUCCESS) {
		cout << "Error creating command queue: " << error << endl;
		exit(error);
	}

	///memory
	float* src_a_h = new float[size];
	float* src_b_h = new float[size];
	float* res_h = new float[size];
	// init vectors
	for (int i = 0; i < size; i++)
	{
		src_a_h[i] = src_a[i];
		src_b_h[i] = src_b[i];
	}
	//allocate device buffer
	const int mem_size = sizeof(float) * size;
	cl_mem src_a_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, (void*)src_a_h, &error);
	cl_mem src_b_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, (void*)src_b_h, &error);
	cl_mem res_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, mem_size, NULL, &error);

	// create the program
	const char* programSource =
		"__kernel void vector_add_gpu(__global const float* src_a, __global const float* src_b, __global float* res, const int num)\n"
		"{\n"
		"	const int idx = get_global_id(0); \n"
		"	float in0 = src_a[idx]; \n"
		"	float in1 = src_b[idx]; \n"
		"	float out = in0 * in1; \n"
		"	res[idx] = out;"
		"}\n";
	cl_program program = clCreateProgramWithSource(context, 1, reinterpret_cast<const char**>(&programSource), NULL, &error);
	if (error != CL_SUCCESS) {
		cout << "Error creating program: " << error << endl;
		exit(error);
	}

	//builds the program
	error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (error != CL_SUCCESS) {
		cout << "Error Build program: " << error << endl;
		// shows the log
		char* build_log;
		size_t log_size;
		// 1st get the log_size
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		build_log = new char[log_size + 1];
		cout << log_size << " log_size" << endl;
		//2nd get log
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
		build_log[log_size] = '\0';
		cout << build_log << " build log" << endl;
		delete[] build_log;
		exit(error);
	}

	// check the kernel code
	size_t bufSize = strlen(programSource) + 1;
	char* programBuffer = (char*)malloc(bufSize);
	size_t program_size_ret;
	error = clGetProgramInfo(program, CL_PROGRAM_SOURCE, bufSize, programBuffer, &program_size_ret);
	if (error != CL_SUCCESS)
	{
		cout << "Error clGetProgramInfo:" << error << endl;
	}

	//extacting the kernel
	cl_kernel vector_add_k = clCreateKernel(program, "vector_add_gpu", &error);
	if (error != CL_SUCCESS) {
		cout << "Error extacting the kernel: " << error << endl;
		exit(error);
	}
	// Enqueuing parameters
	error = clSetKernelArg(vector_add_k, 0, sizeof(cl_mem), &src_a_d);
	if (error != CL_SUCCESS)
	{
		cout << "Error Enqueuing 0 parameters:" << error << endl;
	}
	error |= clSetKernelArg(vector_add_k, 1, sizeof(cl_mem), &src_b_d);
	if (error != CL_SUCCESS)
	{
		cout << "Error Enqueuing 1 parameters:" << error << endl;
	}
	error |= clSetKernelArg(vector_add_k, 2, sizeof(cl_mem), &res_d);
	if (error != CL_SUCCESS)
	{
		cout << "Error Enqueuing 2 parameters:" << error << endl;
	}
	error |= clSetKernelArg(vector_add_k, 3, sizeof(int), &size);
	if (error != CL_SUCCESS)
	{
		cout << "Error Enqueuing 3 parameters:" << error << endl;
	}

	//launching the kernel
	const size_t local_ws = 2;  // Number of work-items per work-group
	const size_t global_ws = shrRoundUp(local_ws, size);
	error = clEnqueueNDRangeKernel(queue, vector_add_k, 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
	if (error != CL_SUCCESS)
	{
		cout << "Error Enqueuing EnqueueNDRangeKernel:" << error << endl;
	}

	// check result
	float* check = new float[size];
	clEnqueueReadBuffer(queue, res_d, CL_TRUE, 0, mem_size, check, 0, NULL, NULL);
	for (int i = 0; i < size; i++)
	{
		res[i] = check[i];
	}

	// Cleaning up
	delete[] src_a_h;
	delete[] src_b_h;
	delete[] res_h;
	delete[] check;
	clReleaseKernel(vector_add_k);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseMemObject(src_a_d);
	clReleaseMemObject(src_b_d);
	clReleaseMemObject(res_d);
}

int main() {

	//////////////////////////////////////////////////////////////// check
	cl_int error = 0;
	bool current = false;
	cl_platform_id current_platform = 0;
	cl_device_id current_device = 0;

	// Platform
	cl_uint num_platforms;
	clGetPlatformIDs(0, NULL, &num_platforms);
	cl_platform_id* platforms = new cl_platform_id[num_platforms];
	clGetPlatformIDs(num_platforms, platforms, NULL);
	for (cl_uint i = 0; i < num_platforms; i++)
	{
		printf("---- %d Platform Name: %s\n", i, getPlatformInfo(platforms[i], CL_PLATFORM_NAME).c_str());
		printf("---- %d Platform Vendor: %s\n", i, getPlatformInfo(platforms[i], CL_PLATFORM_VENDOR).c_str());
		printf("---- %d Platform Version: %s\n", i, getPlatformInfo(platforms[i], CL_PLATFORM_VERSION).c_str());

		// Device
		cl_uint num_devices;
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		cl_device_id* devices = new cl_device_id[num_devices];
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
		for (cl_uint j = 0; j < num_devices; j++)
		{
			// check base
			printf("---- ---- %d Device Name: %s\n", j, getDeviceInfo(devices[j], CL_DEVICE_NAME).c_str());
			printf("---- ---- %d Device Vendor: %s\n", j, getDeviceInfo(devices[j], CL_DEVICE_VENDOR).c_str());
			printf("---- ---- %d Device Version: %s\n", j, getDeviceInfo(devices[j], CL_DEVICE_VERSION).c_str());
			printf("---- ---- %d Driver Version: %s\n", j, getDeviceInfo(devices[j], CL_DRIVER_VERSION).c_str());
			printf("---- ---- %d Device OpenCL C Version: %s\n", j, getDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION).c_str());

			// check type
			std::string device_type = getDeviceInfo(devices[j], CL_DEVICE_TYPE);
			printf("---- ---- %d Device Type: %s\n", j, device_type.c_str());
			if (device_type == "GPU")
			{
				current = true;
				current_platform = platforms[i];
				current_device = devices[j];
			}

			// check fp16
			std::string device_half_preferred_vector_sizes = getDeviceInfo(devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF);
			std::string device_half_native_vector_sizes = getDeviceInfo(devices[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF);
			printf("---- ---- ---- %d Device half Preferred/Native Vector Sizes: %s / %s\n", j, device_half_preferred_vector_sizes.c_str(), device_half_native_vector_sizes.c_str());
			std::string fp16_config = getDeviceInfo(devices[j], CL_DEVICE_SINGLE_FP_CONFIG);
			std::stringstream ss(fp16_config);
			unsigned long long fp_config;
			ss >> fp_config;
			printf("---- ---- ---- %d Device half Denormals:                       %s\n", j, (fp_config & CL_FP_DENORM) > 0 ? "yes" : "no");
			printf("---- ---- ---- %d Device half Infinity and NANs:               %s\n", j, (fp_config & CL_FP_INF_NAN) > 0 ? "yes" : "no");
			printf("---- ---- ---- %d Device half Round to nearest:                %s\n", j, (fp_config & CL_FP_ROUND_TO_NEAREST) > 0 ? "yes" : "no");
			printf("---- ---- ---- %d Device half Round to zero:                   %s\n", j, (fp_config & CL_FP_ROUND_TO_ZERO) > 0 ? "yes" : "no");
			printf("---- ---- ---- %d Device half Round to infinity:               %s\n", j, (fp_config & CL_FP_ROUND_TO_INF) > 0 ? "yes" : "no");
			printf("---- ---- ---- %d Device half IEEE754-2008 fused multiply-add: %s\n", j, (fp_config & CL_FP_FMA) > 0 ? "yes" : "no");
			printf("---- ---- ---- %d Device half Support is emulated in software: %s\n", j, (fp_config & CL_FP_SOFT_FLOAT) > 0 ? "yes" : "no");
		}
	}

	if (!current) {
		printf("!!! No GPU Device !!!\n");
		return 0;
	}

	printf("*** Using Platform: %s ***\n", getPlatformInfo(current_platform, CL_PLATFORM_NAME).c_str());
	printf("*** Using Device: %s ***\n", getDeviceInfo(current_device, CL_DEVICE_NAME).c_str());


	// generate data
	const int size = 100;
	float* a = new float[size];
	float* b = new float[size];
	float* c_cpu_fp32 = new float[size];
	float* c_cl_fp32 = new float[size];
	float* c_cl_fp16 = new float[size];
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			a[i * 10 + j] = j / 10.0;
			b[i * 10 + j] = i / 10.0;
		}
	}

	std::cout << "running with cpu fp32" << std::endl;
	vector_mul_cpu_fp32(a, b, c_cpu_fp32, size);
	std::cout << "running with gpu fp32" << std::endl;
	vector_mul_cl_fp32(current_device, a, b, c_cl_fp32, size);
	std::cout << "running with gpu fp16" << std::endl;
	vector_mul_cl_fp16(current_device, a, b, c_cl_fp16, size);



	delete[] a;
	delete[] b;
	delete[] c_cpu_fp32;
	delete[] c_cl_fp32;
	delete[] c_cl_fp16;


	return 0;




}