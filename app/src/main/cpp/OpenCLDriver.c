#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <android/log.h>
#include <android/bitmap.h>

#include <CL/opencl.h>

#define LOG_TAG "DEBUG"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

#define CL_FILE "/data/local/tmp/sjk_blur.cl"

#define checkCL(expression) {                        \
    cl_int err = (expression);                       \
    if (err < 0 && err > -64) {                      \
        LOGD("Error on line %d. error code: %d\n",   \
                __LINE__, err);                      \
        exit(0);                                     \
    }                                                \
}

JNIEXPORT jobject JNICALL
Java_com_example_opencl_MainActivity_GaussianBlurBitmap(JNIEnv *env, jobject thiz, jobject bitmap) {
    LOGD("reading bitmap info...");
    AndroidBitmapInfo info;
    int ret;
    if ((ret = AndroidBitmap_getInfo(env, bitmap, &info)) < 0) {
        LOGE("AndroidBitmap_getInfo() failed ! error=%d", ret);
        return NULL;
    }
    LOGD("width:%d height:%d stride:%d", info.width, info.height, info.stride);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("Bitmap format is not RGBA_8888!");
        return NULL;
    }

    //read pixels of bitmap into native memory :
    LOGD("reading bitmap pixels...");
    void *bitmapPixels;
    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &bitmapPixels)) < 0) {
        LOGE("AndroidBitmap_lockPixels() failed ! error=%d", ret);
        return NULL;
    }

    uint32_t *src = (uint32_t *) bitmapPixels;
    uint32_t *tempPixels = (uint32_t *) malloc(info.height * info.width * 4);
    int pixelsCount = info.height * info.width;
    memcpy(tempPixels, src, sizeof(uint32_t) * pixelsCount);

    int a, r, g, b;
    float red, green, blue;
    int row, col, cnt = 0;
    int m, n, x, y;
    int width = info.width;
    int height = info.height;

    float mask[9][9] = {
            {0.011237, 0.011637, 0.011931, 0.012111, 0.012172, 0.012111, 0.011931, 0.011637, 0.011237},
            {0.011637, 0.012051, 0.012356, 0.012542, 0.012605, 0.012542, 0.012356, 0.012051, 0.011637},
            {0.011931, 0.012356, 0.012668, 0.012860, 0.012924, 0.012860, 0.012668, 0.012356, 0.011931},
            {0.012111, 0.012542, 0.012860, 0.013054, 0.013119, 0.013054, 0.012860, 0.012542, 0.012111},
            {0.012172, 0.012605, 0.012924, 0.013119, 0.013185, 0.013119, 0.012924, 0.012605, 0.012172},
            {0.012111, 0.012542, 0.012860, 0.013054, 0.013119, 0.013054, 0.012860, 0.012542, 0.012111},
            {0.011931, 0.012356, 0.012668, 0.012860, 0.012924, 0.012860, 0.012668, 0.012356, 0.011931},
            {0.011637, 0.012051, 0.012356, 0.012542, 0.012605, 0.012542, 0.012356, 0.012051, 0.011637},
            {0.011237, 0.011637, 0.011931, 0.012111, 0.012172, 0.012111, 0.011931, 0.011637, 0.011237},
    };

    for (col = 0; col < width; col++) {

        for (row = 0; row < height; row++) {
            blue = 0;
            green = 0;
            red = 0;

            for (m = 0; m < 9; m++) {
                for (n = 0; n < 9; n++) {
                    y = (row + m - 4);
                    x = (col + n - 4);
                    if ((row + m - 4) < 0 || y >= info.height || (col + n - 4) < 0 ||
                        x >= info.width)
                        continue;
                    uint32_t pixel = tempPixels[info.width * y + x];

                    a = (((pixel & (0xff << 24))) >> 24);
                    b = (((pixel & (0xff << 16))) >> 16);
                    g = (((pixel & (0xff << 8))) >> 8);
                    r = (pixel & (0xff));

                    red += r * mask[m][n];
                    green += g * mask[m][n];
                    blue += b * mask[m][n];
                }
            }
            r = (int) red;
            g = (int) green;
            b = (int) blue;
            uint32_t p = (a << 24) + (b << 16) + (g << 8) + (r);
            src[info.width * row + col] = p;
        }
    }

    AndroidBitmap_unlockPixels(env, bitmap);

    free(tempPixels);
    return bitmap;
}

JNIEXPORT jobject JNICALL
Java_com_example_opencl_MainActivity_GaussianBlurGPU(JNIEnv *env, jobject thiz, jobject bitmap) {
    LOGD("reading bitmap info...");
    AndroidBitmapInfo info;
    int ret;
    if ((ret = AndroidBitmap_getInfo(env, bitmap, &info)) < 0) {
        LOGE("AndroidBitmap_getInfo() failed ! error=%d", ret);
        return NULL;
    }
    LOGD("width:%d height:%d stride:%d", info.width, info.height, info.stride);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("Bitmap format is not RGBA_8888!");
        return NULL;
    }

    //read pixels of bitmap into native memory :
    LOGD("reading bitmap pixels...");
    void *bitmapPixels;
    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &bitmapPixels)) < 0) {
        LOGE("AndroidBitmap_lockPixels() failed ! error=%d", ret);
        return NULL;
    }

    uint32_t *src = (uint32_t *) bitmapPixels;
    uint32_t *tempPixels = (uint32_t *) malloc(info.height * info.width * 4);
    int pixelsCount = info.height * info.width;
    memcpy(tempPixels, src, sizeof(uint32_t) * pixelsCount);

    FILE *file_handle;
    char *kernel_file_buffer, *file_log;
    size_t kernel_file_size, log_size;

    unsigned char *cl_file_name = CL_FILE;
    unsigned char *kernel_name = "kernel_blur";

    // Device input buffers
    cl_mem d_src;
    // Device output buffer
    cl_mem d_dst;

    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

    file_handle = fopen(CL_FILE, "r");
    if (file_handle == NULL) {
        printf("Couldn't find the file");
        exit(1);
    }

    //read kernel file
    fseek(file_handle, 0, SEEK_END);
    kernel_file_size = ftell(file_handle);
    rewind(file_handle);
    kernel_file_buffer = (char *) malloc(kernel_file_size + 1);
    kernel_file_buffer[kernel_file_size] = '\0';
    fread(kernel_file_buffer, sizeof(char), kernel_file_size, file_handle);
    fclose(file_handle);

    // Initialize vectors on host
    int i;

    size_t globalSize, localSize, grid;

    // Number of work items in each local work group
    localSize = 64;
    int n_pix = info.width * info.height;

    // Number of total work items - localSize must be devisor
    grid = ((n_pix) % localSize) ? ((n_pix) / localSize) + 1 : (n_pix) / localSize;
    globalSize = grid * localSize;

    cl_int err;

    // Bind to platform
    checkCL(clGetPlatformIDs(1, &cpPlatform, NULL));

    // Get ID for the device
    checkCL(clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL));

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    checkCL(err);

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                                        (const char **) & kernel_file_buffer, &kernel_file_size, &err);
    checkCL(err);

    // Build the program executable
    checkCL(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));

    LOGD("error 22 check");
    checkCL(err);
    if (err != CL_SUCCESS) {
        LOGD("%s", err);
        size_t len;
        char buffer[4096];
        LOGD("Error: Failed to build program executable!");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
                              buffer, &len);

        LOGD("%s", buffer);
        exit(1);
    }
    LOGD("error 323 check");

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, kernel_name, &err);
    checkCL(err);

    // Create the input and output arrays in device memory for our calculation (혹시 에러나면 * 4)
    d_src = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint32_t) * pixelsCount, NULL, &err);
    checkCL(err);
    d_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint32_t) * pixelsCount, NULL, &err);
    checkCL(err);

    // Write our data set into the input array in device memory
    checkCL(clEnqueueWriteBuffer(queue, d_src, CL_TRUE, 0,
                                 sizeof(uint32_t) * pixelsCount, tempPixels, 0, NULL, NULL));

    // Set the arguments to our compute kernel
    checkCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_src));
    checkCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_dst));
    checkCL(clSetKernelArg(kernel, 2, sizeof(uint32_t), &info.width));
    checkCL(clSetKernelArg(kernel, 3, sizeof(uint32_t), &info.height));

    // Execute the kernel over the entire range of the data set
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                   0, NULL, NULL));
    // Wait for the command queue to get serviced before reading back results
    checkCL(clFinish(queue));

    // Read the results from the device
    checkCL(clEnqueueReadBuffer(queue, d_dst, CL_TRUE, 0,
                                sizeof(uint32_t) * pixelsCount, src, 0, NULL, NULL ));

    // release OpenCL resources
    checkCL(clReleaseMemObject(d_src));
    checkCL(clReleaseMemObject(d_dst));
    checkCL(clReleaseProgram(program));
    checkCL(clReleaseKernel(kernel));
    checkCL(clReleaseCommandQueue(queue));
    checkCL(clReleaseContext(context));

    AndroidBitmap_unlockPixels(env, bitmap);

    free(tempPixels);
    return bitmap;
}