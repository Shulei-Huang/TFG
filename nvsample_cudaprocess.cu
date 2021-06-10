#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include <cuda.h>
#include <cudaEGL.h>

#define BOX_W 16
#define BOX_H 8
#define LPF_SIZE 10

typedef uint8_t BYTE;

#define getY(x,y) ((int)(inFrame0[(y)*inPitch+(x)]))
#define sqr(x) ((x)*(x))

__device__ void setPixelNV12(
	BYTE *outFrame0, BYTE *outFrame1, int outPitch, 
	int ox, int oy,
	BYTE vY, BYTE vU, BYTE vV	) {
  BYTE *pOut;
  pOut = outFrame0 + oy * outPitch + ox;
  *pOut = vY;
  pOut = outFrame1 + (oy/2) * outPitch + (ox&~1);
  *pOut = vV;
  pOut++; 
  *pOut = vU; 
}

__global__ void f1Kernel(
		BYTE *inFrame0, BYTE *inFrame1, int inPitch, 
		BYTE *outFrame0, BYTE *outFrame1, int outPitch,
		int width, int height ){
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  // read Y, U, V
  int vY = getY(x,y);
  BYTE *p = inFrame1 + (y/2)*outPitch + (x&~1);
  BYTE vV = p[0];
  BYTE vU = p[1];

  // modify Y
  if(x >= LPF_SIZE && x < width-LPF_SIZE && y >= LPF_SIZE && y < height-LPF_SIZE) {
    int aux=0;
       
    for( int i=-LPF_SIZE; i<=LPF_SIZE; i++ ) {
      for( int j=-LPF_SIZE; j<=LPF_SIZE; j++ ) {
          aux += getY(x+i,y+j);
      }
    }
   
    aux /= sqr(2*LPF_SIZE+1);

    vY -= aux; // sustract low pass filter
    // vY = aux/sqr(2*LPF_SIZE+1);

    vY *= 4; // rescale

    // set vY with constants
    double k1 = -1.3;
    double k2 = 265;
    vY = k1*vY + k2; 

    if( vY < 0 ) vY = 0; else if( vY > 255 ) vY = 255; 	// convert to byte range
  }

  // store Y, U, V
  setPixelNV12(
	outFrame0, outFrame1, outPitch, x, y, (BYTE)vY, vU, vV );

}

static int f1(
     CUeglFrame* eglInputFrame,
     CUeglFrame* eglOutputFrame ) {
    dim3 threadsPerBlock(BOX_W, BOX_H);
    dim3 blocks( eglOutputFrame->width/BOX_W, eglOutputFrame->height/BOX_H );

    f1Kernel<<<blocks,threadsPerBlock>>>( 
		(BYTE *)eglInputFrame->frame.pPitch[0], 
		(BYTE *)eglInputFrame->frame.pPitch[1], eglInputFrame->pitch,
		(BYTE *)eglOutputFrame->frame.pPitch[0], 
		(BYTE *)eglOutputFrame->frame.pPitch[1], eglOutputFrame->pitch,
		eglOutputFrame->width, eglOutputFrame->height
	       	);

    return 0;
}


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
  f1( &eglInputFrame, &eglOutputFrame );

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
