#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cudaEGL.h>

#define BOX_W 16
#define BOX_H 8

__device__ void cudaCopyPixelNV12(
		char *inFrame0, char *inFrame1, int inPitch, 
		char *outFrame0, char *outFrame1, int outPitch,
		int ix, int iy, int ox, int oy ) {
  // read Y, U, V
  char *pIn;
  pIn = inFrame0 + iy * inPitch + ix;
  char vY = *pIn;
  pIn = inFrame1 + (iy/2) * inPitch + (ix&~1);
  char vV = *pIn;
  pIn++; 
  char vU = *pIn; 

  // write Y, U, V
  char *pOut;
  pOut = outFrame0 + oy * outPitch + ox;
  *pOut = vY;
  pOut = outFrame1 + (oy/2) * outPitch + (ox&~1);
  *pOut = vV;
  pOut++; 
  *pOut = vU; 
}

__global__ void cudaMirrorKernel(
		int ax, int ay,
		int bx, int by,
		int cx, int cy,
		double betta, double gamma,
		char *inFrame0, char *inFrame1, int inPitch,
		char *outFrame0, char *outFrame1, int outPitch,
		int outputWidth, int outputHeight ){

  int ox = blockIdx.x*blockDim.x + threadIdx.x;
  int oy = blockIdx.y*blockDim.y + threadIdx.y;

  double x = (double)ox/outputWidth;
  double y = (double)oy/outputHeight;

  double px = (1-x-y)*ax + x*betta*bx + y*gamma*cx;
  double py = (1-x-y)*ay + x*betta*by + y*gamma*cy;
  double div = x*(betta-1) + y*(gamma-1) + 1;

  int ix = (int)round(px/div);
  int iy = (int)round(py/div);

  cudaCopyPixelNV12(
		inFrame0, inFrame1, inPitch,
		outFrame0, outFrame1, outPitch,
		ix, iy, ox, oy );
}

static int cuda_mirror(
     const char *args,
     CUeglFrame* eglInputFrame,
     CUeglFrame* eglOutputFrame ) {

    // parse board positions
    int board_positions[8];
    char *s = strdup( args ); // writable
    int *idx = board_positions;
    for( char *is = strtok( s, "," );
         is != NULL;
         is = strtok( NULL, "," ) ) {
	    *idx = atoi( is );
	    idx++;
    }
    free(s);

  double ax = board_positions[0];
  double ay = board_positions[1];
  double bx = board_positions[2];
  double by = board_positions[3];
  double cx = board_positions[4];
  double cy = board_positions[5];
  double dx = board_positions[6];
  double dy = board_positions[7];

  double a00 = dx-bx;
  double a01 = dx-cx;
  double a10 = dy-by;
  double a11 = dy-cy;
  double b0 = dx-ax;
  double b1 = dy-ay;

  double detm = a00*a11-a01*a10;
  double betta = (a11*b0-a01*b1)/detm;
  double gamma = (-a10*b0+a00*b1)/detm;

    dim3 threadsPerBlock(BOX_W, BOX_H);
    dim3 blocks( eglOutputFrame->width/BOX_W, eglOutputFrame->height/BOX_H ); 
    cudaMirrorKernel<<<blocks,threadsPerBlock>>>( 
		ax, ay, bx, by, cx, cy,
		betta, gamma,
		(char *)eglInputFrame->frame.pPitch[0], 
		(char *)eglInputFrame->frame.pPitch[1], eglInputFrame->pitch,
		(char *)eglOutputFrame->frame.pPitch[0], 
		(char *)eglOutputFrame->frame.pPitch[1], eglOutputFrame->pitch,
		eglOutputFrame->width, eglOutputFrame->height);
    return 0;
}

#if 0
extern "C" 
int do_cuda_process ( int input_dmabuf_fd, int output_dmabuf_fd ) {
  EGLDisplay eglDisplay = eglGetDisplay( EGL_DEFAULT_DISPLAY );
  if (eglDisplay == NULL ) {
    printf("eglGetDisplay failed\n" );
    return 1;
  }

  EGLint major, minor;
  eglInitialize( eglDisplay, &major, &minor );

  EGLImageKHR input_image = NvEGLImageFromFd( eglDisplay, input_dmabuf_fd );
  EGLImageKHR output_image = NvEGLImageFromFd( eglDisplay, output_dmabuf_fd );
#endif

extern "C" 
int do_cuda_process ( const char *args, EGLImageKHR input_image, EGLImageKHR output_image ) {
// NvDestroyEGLImage( NULL, output_image);
// NvDestroyEGLImage( NULL, input_image);

  CUresult status;
  
  cudaFree(0);

  CUgraphicsResource pInputResource = NULL;
  status = cuGraphicsEGLRegisterImage(&pInputResource, input_image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
  if (status != CUDA_SUCCESS) {
    printf("cuGraphicsEGLRegisterImage failed : %d \n", status);
    return 1;
  }

  CUeglFrame eglInputFrame;
  status = cuGraphicsResourceGetMappedEglFrame( &eglInputFrame, pInputResource, 0, 0);
  if (status != CUDA_SUCCESS) {
    printf ("cuGraphicsSubResourceGetMappedArray failed\n");
    return 1;
  }

  CUgraphicsResource pOutputResource = NULL;
  status = cuGraphicsEGLRegisterImage(&pOutputResource, output_image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
  if (status != CUDA_SUCCESS) {
    printf("cuGraphicsEGLRegisterImage failed : %d \n", status);
    return 1;
  }

  CUeglFrame eglOutputFrame;
  status = cuGraphicsResourceGetMappedEglFrame( &eglOutputFrame, pOutputResource, 0, 0);
  if (status != CUDA_SUCCESS) {
    printf ("cuGraphicsSubResourceGetMappedArray failed\n");
    return 1;
  }

#if 0
  printf( "width: %d %d\n", eglInputFrame.width, eglOutputFrame.width );
  printf( "height: %d %d\n", eglInputFrame.height, eglOutputFrame.height );
  printf( "planeCount: %d %d\n", eglInputFrame.planeCount, eglOutputFrame.planeCount );
  printf( "frame type: %s %s\n", 
		  eglInputFrame.frameType == CU_EGL_FRAME_TYPE_ARRAY ? "array" : "pitch", 
		  eglOutputFrame.frameType == CU_EGL_FRAME_TYPE_ARRAY ? "array" : "pitch" );
  printf( "numChannels: %d %d\n", eglInputFrame.numChannels, eglOutputFrame.numChannels );
  printf( "pitch: %d %d\n", eglInputFrame.pitch, eglOutputFrame.pitch );
  printf( "CUeglColorFormat: %d %d\n", eglInputFrame.eglColorFormat, eglOutputFrame.eglColorFormat );
  printf( "depth: %d %d\n", eglInputFrame.depth, eglOutputFrame.depth );
  printf( "CUarray_format: %d %d\n", eglInputFrame.cuFormat, eglOutputFrame.cuFormat );
#endif

  status = cuCtxSynchronize();
  if (status != CUDA_SUCCESS) {
    printf ("cuCtxSynchronize failed \n");
    return 1;
  }

//  if (eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
//    if (eglFrame.eglColorFormat == CU_EGL_COLOR_FORMAT_ABGR) {
  cuda_mirror( args, &eglInputFrame, &eglOutputFrame );

  status = cuCtxSynchronize();
  if (status != CUDA_SUCCESS) {
    printf ("cuCtxSynchronize failed after cuda_mirror \n");
    return 1;
  }

  status = cuGraphicsUnregisterResource(pInputResource);
  if (status != CUDA_SUCCESS) {
    printf("cuGraphicsEGLUnRegisterResource failed: %d \n", status);
    return 1;
  }

  status = cuGraphicsUnregisterResource(pOutputResource);
  if (status != CUDA_SUCCESS) {
    printf("cuGraphicsEGLUnRegisterResource failed: %d \n", status);
    return 1;
  }

  return 0;
}
