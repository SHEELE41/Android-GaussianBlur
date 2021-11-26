//OpenCL kernel. Each work item takes care of one element of c
 
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
 
#define RGB8888_A(p) ((p & (0xff<<24))      >> 24 )
#define RGB8888_B(p) ((p & (0xff << 16)) >> 16 )
#define RGB8888_G(p) ((p & (0xff << 8))  >> 8 )
#define RGB8888_R(p) (p & (0xff) )
 
__kernel void kernel_blur(__global int *src, __global int *dst, const int width, const int height){
    //int id = get_global_id(0);
    
    int row = get_global_id(0)/width;
    int col = get_global_id(0)%width;
 
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
 
    int a,r, g, b;
 
    float red = 0, green = 0, blue = 0;
    int m, n,x,y;
 
    for (m = 0; m < 9; m++) {
        for (n = 0; n < 9; n++) {
            y = (row + m - 4);
            x = (col + n - 4);
            if ((row + m - 4) < 0 || y >= height || (col + n - 4) < 0 || x >= width) continue;
            int pixel = src[width * y + x];
 
            //
            a = RGB8888_A(pixel);
            r = RGB8888_R(pixel);
            g = RGB8888_G(pixel);
            b = RGB8888_B(pixel);
 
 
            red += r * mask[m][n];
            green += g * mask[m][n];
            blue += b * mask[m][n];
        }
    }
    r = (int) red;
    g = (int) green;
    b = (int) blue;
 
    int v = (a << 24) + (b << 16) + (g << 8) + (r);
    dst[width * row + col] = v;
}

