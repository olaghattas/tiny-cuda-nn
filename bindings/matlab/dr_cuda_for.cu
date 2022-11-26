#include <cuda.h>
#include <cuda_runtime.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"

// define consts
#define eps 1e-15

__global__ void dr_cuda_forward_render_batch(
        const float *__restrict__ points3d_bxfx9,
        const float *__restrict__ points2d_bxfx6,
        const float *__restrict__ pointsdirect_bxfx1,
        const float *__restrict__ pointsbbox_bxfx4,
        const float *__restrict__ colors_bxfx3d,
        int *__restrict__ imidx_bxhxwx1,
        float *__restrict__ imwei_bxhxwx3,
        float *__restrict__ im_bxhxwxd,
        float *__restrict__ pixel_covered_list,
        int *__restrict__ pixel_covered_ind_list,
        int *__shared__ pixel_covered_list_len,
        int *__restrict__ pixel_covered,
        const int batch_size, const int height, const int width, const int fnum,
        const int dnum) {


    int presentthread = blockIdx.x * blockDim.x + threadIdx.x;
    int wididx = presentthread % width;
    presentthread = (presentthread - wididx) / width;
    int heiidx = presentthread % height;
    int bidx = (presentthread - heiidx) / height;

    if (bidx >= batch_size || heiidx >= height || wididx >= width) {
        return;
    }
    // which pixel it belongs to
    const int batchOffsetIdx = bidx*height*width;
    const int pixelIdx = wididx*height+heiidx;
    const int totalidx1 = batchOffsetIdx + pixelIdx;
    const int numPixels = height*width;
//    const int totalidx3 = totalidx1 * 3;
//    const int totalidxd = totalidx1 * dnum;


    // pixel coordinate
    float x0 = 1.0 / width * (2 * wididx + 1 - width);
    float y0 = 1.0 / height * (height - 2 * heiidx - 1);
    // init depth buffer
    float znow = 1;

    for (int fidxint = 0; fidxint < fnum; fidxint++) {
        // which face it belongs to
        const int shift1 = bidx * fnum + fidxint;
        const int shift4 = shift1 * 4;
        const int shift6 = shift1 * 6;
        const int shift9 = shift1 * 9;
        const int shift3d = shift1 * 3 * dnum;

        // is this face visible?
        float direction = pointsdirect_bxfx1[shift1];
        if (direction < 0) {
            continue;
        }

        ///////////////////////////////////////////////////////////////
        // will this pixel is influenced by this face?
        float xmin = pointsbbox_bxfx4[shift4 + 0];
        float ymin = pointsbbox_bxfx4[shift4 + 1];
        float xmax = pointsbbox_bxfx4[shift4 + 2];
        float ymax = pointsbbox_bxfx4[shift4 + 3];

        // not covered by this face!
        if (x0 < xmin || x0 >= xmax || y0 < ymin || y0 >= ymax) {
            continue;
        }

        //////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
        // if this pixel is covered by this face, then we check its depth and weights
        float ax = points2d_bxfx6[shift6 + 0];
        float ay = points2d_bxfx6[shift6 + 1];
        float bx = points2d_bxfx6[shift6 + 2];
        float by = points2d_bxfx6[shift6 + 3];
        float cx = points2d_bxfx6[shift6 + 4];
        float cy = points2d_bxfx6[shift6 + 5];

        // replace with other variables
        float m = bx - ax;
        float p = by - ay;

        float n = cx - ax;
        float q = cy - ay;

        float s = x0 - ax;
        float t = y0 - ay;


        float k1 = s * q - n * t;
        float k2 = m * t - s * p;
        float k3 = m * q - n * p;

        float w1 = k1 / (k3 + eps);
        float w2 = k2 / (k3 + eps);
        float w0 = 1 - w1 - w2;

        // not lie in the triangle
        if (w0 < 0 || w1 < 0 || w2 < 0) {
            continue;
        }

        //////////////////////////////////////////////////////////////////////////////////////
        // if it is perspective, then this way has a little error
        // because face plane may not be parallel to the image plane
        // but let's ignore it first
        float az = points3d_bxfx9[shift9 + 2];
        float bz = points3d_bxfx9[shift9 + 5];
        float cz = points3d_bxfx9[shift9 + 8];

        float z0 = w0 * az + w1 * bz + w2 * cz;

        // it will be filled by a nearer face
        if (z0 <= znow && znow < 0) {
            continue;
        }
        // update  depth
        znow = z0;
        // index
        imidx_bxhxwx1[totalidx1] = fidxint;
        pixel_covered[totalidx1] = 1;
        // wei
        imwei_bxhxwx3[batchOffsetIdx*3 + pixelIdx + 0*numPixels] = w0;
        imwei_bxhxwx3[batchOffsetIdx*3 + pixelIdx + 1*numPixels] = w1;
        imwei_bxhxwx3[batchOffsetIdx*3 + pixelIdx + 2*numPixels] = w2;
        // color
        for (int d = 0; d < dnum; d++) {
            float r0 = colors_bxfx3d[shift3d + d];
            float r1 = colors_bxfx3d[shift3d + dnum + d];
            float r2 = colors_bxfx3d[shift3d + dnum + dnum + d];
            im_bxhxwxd[batchOffsetIdx*dnum + pixelIdx + d*numPixels] = w0 * r0 + w1 * r1 + w2 * r2;
        }
    }
    // if the pixel is covered, then add to list
    if (pixel_covered[totalidx1] == 1) {
        int val = atomicAdd(&pixel_covered_list_len[bidx], 1);
        pixel_covered_list[2*numPixels*bidx + 2*val] = x0;
        pixel_covered_list[2*numPixels*bidx +  2*val+1] = y0;
        pixel_covered_ind_list[2*numPixels*bidx + 2*val+0] = wididx;
        pixel_covered_ind_list[2*numPixels*bidx + 2*val+1] = heiidx;
    }
}

__device__ void dr_cuda_forward_prob_batch2(
        const float *__restrict__ points3d_bxfx9,
        const float *__restrict__  points2d_bxfx6,
        const int *__restrict__ imidx_bxhxwx1,
        const float * __restrict__ imwei_bxhxwx3,
        const float *__restrict__ pointsdirect_bxfx1,
        const float *__restrict__ pixel_covered_list,
        const int *__restrict__ pixel_covered_ind_list,
        const int *__restrict__ pixel_covered_list_len,
        const int *__restrict__ pixel_covered,
        float *__restrict__ closest_point_bxhxwx2,
        float *__restrict__ closest_point_bxhxwx3,
        int *__restrict__ closest_face_bxhxwx1,
        float *__restrict__ improb_bxhxwx1,
        const int batch_size, const int height, const int width, const int fnum) {

    int presentthread = blockIdx.x * blockDim.x + threadIdx.x;
    int wididx = presentthread % width;
    presentthread = (presentthread - wididx) / width;
    int heiidx = presentthread % height;
    int bidx = (presentthread - heiidx) / height;

    if (bidx >= batch_size || heiidx >= height || wididx >= width) {
        return;
    }
    float x0 = 1.0 / width * (2 * wididx + 1 - width);
    float y0 = 1.0 / height * (height - 2 * heiidx - 1);
    const int batchOffsetIdx = bidx*height*width;
    const int pixelIdx = wididx*height+heiidx;
    const int totalidx1 = batchOffsetIdx + pixelIdx;
    const int numPixels = height*width;
    improb_bxhxwx1[totalidx1] = -1;

    for (int fidxint = 0; fidxint < fnum; fidxint++) {
        // which face it belongs to
        const int shift1 = bidx * fnum + fidxint;
        const int shift6 = shift1 * 6;

        // is this face visible?
        float direction = pointsdirect_bxfx1[shift1];
        if (direction < 0) {
            continue;
        }
        const float* vert2d = &points2d_bxfx6[shift6];
        for (int i = 0; i < 3; i++){
            float xi = vert2d[2*i+0];
            float yi = vert2d[2*i+1];
            float dissquare = (xi - x0) * (xi - x0) + (yi - y0) * (yi - y0);
            if (improb_bxhxwx1[totalidx1] > dissquare || improb_bxhxwx1[totalidx1] < 0) {
                improb_bxhxwx1[totalidx1] = dissquare;
                closest_face_bxhxwx1[totalidx1] = fidxint;
                closest_point_bxhxwx2[batchOffsetIdx*2 + pixelIdx + 0*numPixels] = xi;
                closest_point_bxhxwx2[batchOffsetIdx*2 + pixelIdx + 1*numPixels] = yi;
                const int shift1 = bidx * fnum + fidxint;
                const int shift9 = shift1 * 9;
                closest_point_bxhxwx3[totalidx1 * 3 + 0] = points3d_bxfx9[shift9 + 3*i + 0];
                closest_point_bxhxwx3[totalidx1 * 3 + 1] = points3d_bxfx9[shift9 + 3*i + 1];
                closest_point_bxhxwx3[totalidx1 * 3 + 2] = points3d_bxfx9[shift9 + 3*i + 2];
            }
        }
    }
}

__global__ void dr_cuda_forward_prob_batch(
        const float *__restrict__ points3d_bxfx9,
        const float *__restrict__  points2d_bxfx6,
        const int *__restrict__ imidx_bxhxwx1,
        const float * __restrict__ imwei_bxhxwx3,
        const float *__restrict__ pointsdirect_bxfx1,
        const float *__restrict__ pixel_covered_list,
        const int *__restrict__ pixel_covered_ind_list,
        const int *__restrict__ pixel_covered_list_len,
        const int *__restrict__ pixel_covered,
        float *__restrict__ closest_point_bxhxwx2,
        float *__restrict__ closest_point_bxhxwx3,
        int *__restrict__ closest_face_bxhxwx1,
        float *__restrict__ improb_bxhxwx1,
        const int batch_size, const int height, const int width, const int fnum) {

    int presentthread = blockIdx.x * blockDim.x + threadIdx.x;
    int wididx = presentthread % width;
    presentthread = (presentthread - wididx) / width;
    int heiidx = presentthread % height;
    int bidx = (presentthread - heiidx) / height;

    if (bidx >= batch_size || heiidx >= height || wididx >= width) {
        return;
    }
    if (pixel_covered_list_len[bidx] == 0){
        dr_cuda_forward_prob_batch2(points3d_bxfx9, points2d_bxfx6,
        imidx_bxhxwx1,imwei_bxhxwx3,pointsdirect_bxfx1,pixel_covered_list, pixel_covered_ind_list,
        pixel_covered_list_len,pixel_covered,closest_point_bxhxwx2,closest_point_bxhxwx3,
        closest_face_bxhxwx1,improb_bxhxwx1,batch_size, height, width, fnum);
        return;
    }
    // pixel coordinate
    float x0 = 1.0 / width * (2 * wididx + 1 - width);
    float y0 = 1.0 / height * (height - 2 * heiidx - 1);
    // which pixel it belongs to
    const int batchOffsetIdx = bidx*height*width;
    const int pixelIdx = wididx*height+heiidx;
    const int totalidx1 = batchOffsetIdx + pixelIdx;
    const int numPixels = height*width;


    improb_bxhxwx1[totalidx1] = -1; // init

    int closest_point_idx = -1;
    float xi = -1;
    float yi = -1;

    if (pixel_covered[totalidx1] == 1) {
        improb_bxhxwx1[totalidx1] = 0.0;
        xi = x0;
        yi = y0;
        closest_point_idx = pixelIdx;
    } else { //  pixels not covered by any faces
        for (int i = 0; i < pixel_covered_list_len[bidx]; i++) {
            xi = pixel_covered_list[2*numPixels*bidx + 2 * i];
            yi = pixel_covered_list[2*numPixels*bidx + 2 * i + 1];
            float dissquare = (xi - x0) * (xi - x0) + (yi - y0) * (yi - y0);
            if (improb_bxhxwx1[totalidx1] > dissquare || improb_bxhxwx1[totalidx1] < 0) {
                improb_bxhxwx1[totalidx1] = dissquare;
                int wididx2 = pixel_covered_ind_list[2*numPixels*bidx + 2 * i];
                int heiidx2 = pixel_covered_ind_list[2*numPixels*bidx + 2 * i + 1];
                closest_point_idx = wididx2 * height + heiidx2;
            }
        }

    }
    closest_face_bxhxwx1[totalidx1] = imidx_bxhxwx1[batchOffsetIdx + closest_point_idx];

    closest_point_bxhxwx2[batchOffsetIdx*2 + pixelIdx + 0*numPixels] = xi;
    closest_point_bxhxwx2[batchOffsetIdx*2 + pixelIdx + 1*numPixels] = yi;

    float w0 = imwei_bxhxwx3[batchOffsetIdx*3 + closest_point_idx + 0*numPixels];
    float w1 = imwei_bxhxwx3[batchOffsetIdx*3 + closest_point_idx + 1*numPixels];
    float w2 = imwei_bxhxwx3[batchOffsetIdx*3 + closest_point_idx + 2*numPixels];
    const int faceIdx = closest_face_bxhxwx1[totalidx1];
    const int shift9 = (fnum*bidx + faceIdx)*9;
    float ax = points3d_bxfx9[shift9 + 0];
    float bx = points3d_bxfx9[shift9 + 3];
    float cx = points3d_bxfx9[shift9 + 6];
    float ay = points3d_bxfx9[shift9 + 1];
    float by = points3d_bxfx9[shift9 + 4];
    float cy = points3d_bxfx9[shift9 + 7];
    float az = points3d_bxfx9[shift9 + 2];
    float bz = points3d_bxfx9[shift9 + 5];
    float cz = points3d_bxfx9[shift9 + 8];

    float point_x = w0 * ax + w1 * bx + w2 * cx;
    float point_y = w0 * ay + w1 * by + w2 * cy;
    float point_z = w0 * az + w1 * bz + w2 * cz;
    closest_point_bxhxwx3[batchOffsetIdx*3 + pixelIdx + 0*numPixels] = point_x;
    closest_point_bxhxwx3[batchOffsetIdx*3 + pixelIdx + 1*numPixels] = point_y;
    closest_point_bxhxwx3[batchOffsetIdx*3 + pixelIdx + 2*numPixels] = point_z;
}





mxGPUArray *getMXGPU(int size, mxClassID type) {
    mwSize dims[1];
    *(dims) = size;
    return mxGPUCreateGPUArray(1, dims, type, mxREAL, MX_GPU_INITIALIZE_VALUES);
 }

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]) {
    // Perform error checks//    /
    // Initialize the MathWorks GPU API
    mxInitGPU();
    if (nrhs != 10) {
        char const *const errId = "parallel:gpu:mexGPUExample:InvalidInput";
        char const *const errMsg = "Wrong number of inputs";
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    for (int i = 0; i < 5; i++) {
        if (!(mxIsGPUArray(prhs[i]))) {
            char const *const errId = "parallel:gpu:mexGPUExample:InvalidInput";
            char const *const errMsg = "Not all inputs are on the GPU";
            mexErrMsgIdAndTxt(errId, errMsg);
        }
    }

    // Declare all read variables.
    mxGPUArray const *tmp1 = mxGPUCreateFromMxArray(prhs[0]);
    float const *points3d_bxfx9 = (float const *) (mxGPUGetDataReadOnly(tmp1));
    mxGPUArray const *tmp2 = mxGPUCreateFromMxArray(prhs[1]);
    float const *points2d_bxfx6 = (float const *) (mxGPUGetDataReadOnly(tmp2));
    mxGPUArray const *tmp3 = mxGPUCreateFromMxArray(prhs[2]);
    float const *pointsdirect_bxfx1 = (float const *) (mxGPUGetDataReadOnly(tmp3));
    mxGPUArray const *tmp4 = mxGPUCreateFromMxArray(prhs[3]);
    float const *pointsbbox_bxfx4 = (float const *) (mxGPUGetDataReadOnly(tmp4));
    mxGPUArray const *tmp5 = mxGPUCreateFromMxArray(prhs[4]);
    float const *colors_bxfx3d = (float const *) (mxGPUGetDataReadOnly(tmp5));
    int height = *((double *) mxGetData(prhs[5]));
    int width = *((double *) mxGetData(prhs[6]));
    int dnum = *((double *) mxGetData(prhs[7]));
    int fnum = *((double *) mxGetData(prhs[8]));
    int batch_size = *((double *) mxGetData(prhs[9]));

    int num_pixels = height * width;
    // Declare all created variables.
    mxGPUArray *im_bxhxwxdMX = getMXGPU(batch_size*dnum * num_pixels, mxSINGLE_CLASS);
    mxGPUArray *improb_bxhxwx1MX = getMXGPU(batch_size*num_pixels, mxSINGLE_CLASS);
    mxGPUArray *imwei_bxhxwx3MX = getMXGPU(batch_size*3 * num_pixels, mxSINGLE_CLASS);
    mxGPUArray *closest_point_bxhxwx2MX = getMXGPU(batch_size*2 * num_pixels, mxSINGLE_CLASS);
    mxGPUArray *closest_point_bxhxwx3MX = getMXGPU(batch_size*3 * num_pixels, mxSINGLE_CLASS);
    mxGPUArray *closest_face_bxhxwx1MX = getMXGPU(batch_size*num_pixels, mxINT32_CLASS);
    mxGPUArray *pixel_coveredMX = getMXGPU(batch_size*num_pixels, mxINT32_CLASS);
    mxGPUArray *imidx_bxhxwx1MX = getMXGPU(batch_size*num_pixels, mxINT32_CLASS);
//    mxGPUArray* pixel_covered_listMX = getMXGPU(2*num_pixels, mxINT32_CLASS);

    float *im_bxhxwxd = (float *) (mxGPUGetData(im_bxhxwxdMX));
    float *improb_bxhxwx1 = (float *) (mxGPUGetData(improb_bxhxwx1MX));
    float *imwei_bxhxwx3 = (float *) (mxGPUGetData(imwei_bxhxwx3MX));
    float *closest_point_bxhxwx2 = (float *) (mxGPUGetData(closest_point_bxhxwx2MX));
    float *closest_point_bxhxwx3 = (float *) (mxGPUGetData(closest_point_bxhxwx3MX));
    int *closest_face_bxhxwx1 = (int *) (mxGPUGetData(closest_face_bxhxwx1MX));
    int *pixel_covered = (int *) (mxGPUGetData(pixel_coveredMX));
    // allocate intermediate variables
    int *imidx_bxhxwx1 = (int *) (mxGPUGetData(imidx_bxhxwx1MX));

    // allocate raw cuda variables
    float *pixel_covered_list;
    cudaMalloc(&pixel_covered_list, batch_size * 2 * num_pixels * sizeof(float));
    int* pixel_covered_ind_list;
    cudaMalloc(&pixel_covered_ind_list, batch_size *2 * num_pixels * sizeof(int));
    int *pixel_covered_list_len;
    cudaMalloc(&pixel_covered_list_len, batch_size * sizeof(int));
    int h_pixel_covered_list_len = 0;
    for (int b = 0; b < batch_size; b++){
        cudaMemcpy(&pixel_covered_list_len[b], &h_pixel_covered_list_len, sizeof(int), cudaMemcpyHostToDevice);
    }
    // thread setup
    const int threadnum = 1024;
    const int totalthread = batch_size * height * width;
    const int blocknum = totalthread / threadnum + 1;
    const dim3 threads(threadnum, 1, 1);
    const dim3 blocks(blocknum, 1, 1);

    dr_cuda_forward_render_batch<<<blocks, threads>>>(
            points3d_bxfx9, points2d_bxfx6, pointsdirect_bxfx1,
            pointsbbox_bxfx4, colors_bxfx3d, imidx_bxhxwx1,
            imwei_bxhxwx3, im_bxhxwxd,
            pixel_covered_list, pixel_covered_ind_list, pixel_covered_list_len, pixel_covered,
            batch_size, height, width, fnum, dnum);

//    cudaDeviceSynchronize();
//    cudaMemcpy(&h_pixel_covered_list_len, pixel_covered_list_len, sizeof(int), cudaMemcpyDeviceToHost);
//    printf("num points:  %d \n", h_pixel_covered_list_len);
//int* h_pixel_covered_list = (int*) malloc(2*num_pixels*sizeof(int));
//cudaMemcpy(h_pixel_covered_list, pixel_covered_list, 2*num_pixels*sizeof(int), cudaMemcpyDeviceToHost);
//for(int i =0; i< h_pixel_covered_list_len;i++){
//    printf("pixel point:  (%d, %d) \n", h_pixel_covered_list[2*i], h_pixel_covered_list[2*i+1]);
//}
//    if (h_pixel_covered_list_len > 0) {
        dr_cuda_forward_prob_batch<<<blocks, threads>>>(
                points3d_bxfx9,
                points2d_bxfx6,
                imidx_bxhxwx1,
                imwei_bxhxwx3,
                pointsdirect_bxfx1,
                pixel_covered_list,
                pixel_covered_ind_list,
                pixel_covered_list_len,
                pixel_covered,
                closest_point_bxhxwx2,
                closest_point_bxhxwx3,
                closest_face_bxhxwx1,
                improb_bxhxwx1,
                batch_size, height, width, fnum);
//    } else{
//        dr_cuda_forward_prob_batch2<<<blocks, threads>>>(
//                points3d_bxfx9,
//                points2d_bxfx6,
//                imidx_bxhxwx1,
//                imwei_bxhxwx3,
//                pointsdirect_bxfx1,
//                pixel_covered_list,
//                pixel_covered_ind_list,
//                pixel_covered_list_len,
//                pixel_covered,
//                closest_point_bxhxwx2,
//                closest_point_bxhxwx3,
//                closest_face_bxhxwx1,
//                improb_bxhxwx1,
//                batch_size, height, width, fnum);
//    }

    mwSize im_bxhxwxdDims[4] = {(mwSize)height, (mwSize)width, (mwSize) dnum, (mwSize)batch_size};
    mxGPUSetDimensions(im_bxhxwxdMX, im_bxhxwxdDims, 4);

    mwSize improb_bxhxwx1Dims[4] = {(mwSize)height, (mwSize)width, (mwSize) 1, (mwSize)batch_size};
    mxGPUSetDimensions(improb_bxhxwx1MX, improb_bxhxwx1Dims, 4);

    mwSize imwei_bxhxwx3Dims[4] = {(mwSize)height, (mwSize)width, (mwSize) 3, (mwSize)batch_size};
    mxGPUSetDimensions(imwei_bxhxwx3MX, imwei_bxhxwx3Dims, 4);

    mwSize closest_point_bxhxwx2Dims[4] = {(mwSize)height, (mwSize)width, (mwSize) 2, (mwSize)batch_size};
    mxGPUSetDimensions(closest_point_bxhxwx2MX, closest_point_bxhxwx2Dims, 4);

    mwSize closest_point_bxhxwx3Dims[4] = {(mwSize)height, (mwSize)width, (mwSize) 3, (mwSize)batch_size};
    mxGPUSetDimensions(closest_point_bxhxwx3MX, closest_point_bxhxwx3Dims, 4);

    mwSize closest_face_bxhxwx1Dims[4] = {(mwSize)height, (mwSize)width, (mwSize) 1, (mwSize)batch_size};
    mxGPUSetDimensions(closest_face_bxhxwx1MX, closest_face_bxhxwx1Dims, 4);

    mwSize pixel_coveredDims[4] = {(mwSize)height, (mwSize)width, (mwSize) 1, (mwSize)batch_size};
    mxGPUSetDimensions(pixel_coveredMX, pixel_coveredDims, 4);

//    printf("dims: [%z %z %z %z]",dims);

    plhs[0] = mxGPUCreateMxArrayOnGPU(im_bxhxwxdMX);
    plhs[1] = mxGPUCreateMxArrayOnGPU(improb_bxhxwx1MX);
    plhs[2] = mxGPUCreateMxArrayOnGPU(imwei_bxhxwx3MX);
    plhs[3] = mxGPUCreateMxArrayOnGPU(closest_point_bxhxwx2MX);
    plhs[4] = mxGPUCreateMxArrayOnGPU(closest_point_bxhxwx3MX);
    plhs[5] = mxGPUCreateMxArrayOnGPU(closest_face_bxhxwx1MX);
    plhs[6] = mxGPUCreateMxArrayOnGPU(pixel_coveredMX);

    mxGPUDestroyGPUArray(pixel_coveredMX);
    mxGPUDestroyGPUArray(im_bxhxwxdMX);
    mxGPUDestroyGPUArray(improb_bxhxwx1MX);
    mxGPUDestroyGPUArray(imwei_bxhxwx3MX);
    mxGPUDestroyGPUArray(closest_point_bxhxwx2MX);
    mxGPUDestroyGPUArray(closest_point_bxhxwx3MX);
    mxGPUDestroyGPUArray(closest_face_bxhxwx1MX);
    mxGPUDestroyGPUArray(imidx_bxhxwx1MX);

    mxGPUDestroyGPUArray(tmp1);
    mxGPUDestroyGPUArray(tmp2);
    mxGPUDestroyGPUArray(tmp3);
    mxGPUDestroyGPUArray(tmp4);
    mxGPUDestroyGPUArray(tmp5);

    cudaFree(pixel_covered_list_len);
    cudaFree(pixel_covered_list);
    cudaFree(pixel_covered_ind_list);
}

