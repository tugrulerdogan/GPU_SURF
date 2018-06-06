#include <stdint.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define CHANNEL_NUM 3

const int N = 16; 
const int blocksize = 16; 


__global__ void convert_to_grayscale(uint8_t *a, uint8_t *b) 
{

    int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int pos = threadId;
    int g_pos = pos * 3;

    // RGB to monochrome formula: (0.2125 * red) + (0.7154 * green) + (0.0721 * blue)
    b[pos] = a[g_pos+1] * 0.2125; //RED
    b[pos] += a[g_pos+2] * 0.7154; //GREEN
    b[pos] += a[g_pos+3] * 0.0721; //BLUE


}
 


__global__ void hessian(uint8_t const  __restrict__ *b, uint8_t *h, int8_t const  __restrict__ *f1){


    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int sums = 0;  // will be private type
    //TODO: width height 4er piksel daha dar verilmeli
    int width = f1[0];  // will be private type
    int heigth = f1[1];  // will be private type

    int pos = threadId + 4 * width + 4; // will be private type
    int f_pos = 2; // will be private type
    int pos_buffer = pos; // will be private type

    for (int order =0; order < 4 ; order++ ){

        sums =0;
    
        for (int i =0; i< 9; i++){
            for(int j =0; j< 9; j++){
    
               // forward to the next pixel
               sums += b[pos_buffer + j] * f1[f_pos++];
    
    
            }
    
            // go to the bottom line of the photo 
            pos_buffer += width;
    
        }
    
        // normalize filter result by filter coefficients
        h[pos*4 + order] = sums / f1[f_pos++];


    }
}


int main() {


    int width, height, bpp;

    float elapsed=0;
    cudaEvent_t start, stop;

    uint8_t* rgb_image = stbi_load("1.jpg", &width, &height, &bpp, 3);

    int8_t f1[324] ={ 0,0,1,1,1,1,1,0,0,
                     0,0,1,1,1,1,1,0,0,
                     0,0,1,1,1,1,1,0,0,
                     0,0,-2,-2,-2,-2,-2,0,0,
                     0,0,-2,-2,-2,-2,-2,0,0,
                     0,0,-2,-2,-2,-2,-2,0,0,
                     0,0,1,1,1,1,1,0,0,
                     0,0,1,1,1,1,1,0,0,
                     0,0,1,1,1,1,1,0,0,
                     0,0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0,0,
                     1,1,1,-2,-2,-2,1,1,1,
                     1,1,1,-2,-2,-2,1,1,1,
                     1,1,1,-2,-2,-2,1,1,1,
                     1,1,1,-2,-2,-2,1,1,1,
                     1,1,1,-2,-2,-2,1,1,1,
                     0,0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0,0,
                     0,1,1,1,0,-1,-1,-1,0,
                     0,1,1,1,0,-1,-1,-1,0,
                     0,1,1,1,0,-1,-1,-1,0,
                     0,0,0,0,0,0,0,0,0,
                     0,-1,-1,-1,0,1,1,1,0,
                     0,-1,-1,-1,0,1,1,1,0,
                     0,-1,-1,-1,0,1,1,1,0,
                     0,0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0,0,
                     0,-1,-1,-1,0,1,1,1,0,
                     0,-1,-1,-1,0,1,1,1,0,
                     0,-1,-1,-1,0,1,1,1,0,
                     0,0,0,0,0,0,0,0,0,
                     0,1,1,1,0,-1,-1,-1,0,
                     0,1,1,1,0,-1,-1,-1,0,
                     0,1,1,1,0,-1,-1,-1,0,
                     0,0,0,0,0,0,0,0,0,
                    };


    int im_size = width * height;

    uint8_t* return_rgb_image = (uint8_t*)malloc(im_size * 4);


 
    uint8_t *c_image;
    uint8_t *return_image;
    uint8_t *hessian_image;
    int8_t *g_f1;
 
 
    cudaMalloc( (void**)&c_image, im_size * CHANNEL_NUM ); 
    cudaMalloc( (void**)&return_image, im_size); 
    cudaMalloc( (void**)&hessian_image, im_size*4); 
    cudaMalloc( (void**)&g_f1, 324); 

    dim3 dimBlock( blocksize, blocksize );
    dim3 dimGrid( (width-4)/blocksize, (height-4)/blocksize );

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i =0 ; i< 60; i++){



        cudaMemcpy( c_image, rgb_image, im_size * CHANNEL_NUM, cudaMemcpyHostToDevice ); 
        cudaMemcpy( g_f1, f1, 81, cudaMemcpyHostToDevice ); 

        convert_to_grayscale<<<dimGrid, dimBlock>>>(c_image, return_image);
        hessian<<<dimGrid, dimBlock>>>(return_image, hessian_image, g_f1);


        cudaMemcpy( return_rgb_image, hessian_image, im_size*4, cudaMemcpyDeviceToHost ); 
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop) ;

    cudaEventElapsedTime(&elapsed, start, stop) ;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("time: %.2f ms\n", elapsed);

    cudaFree( c_image );
    cudaFree( return_image );
    cudaFree( hessian_image );
    
    stbi_write_png("image.png", width, height, 1, return_rgb_image, width);
    stbi_image_free(rgb_image);
    stbi_image_free(return_rgb_image);

    return EXIT_SUCCESS;
}
