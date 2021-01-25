w//   NBK s185205 02346, F20, DTU
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#include <stdint.h>

#include <string.h>
#include <omp.h>

#include "readBMP.h"
#include "writeBMP.h"

#define MASTER 0
#define WINDOW_SIZE 5
void createImageArray(char *, PIXEL_ARRAY *);
void averagingFilterNaive(PIXEL_ARRAY *, PIXEL_ARRAY *, int);
void mpiDistribute(PIXEL_ARRAY *, PIXEL_ARRAY *, int, int, int, int);
void averagingFilterOpenMP(PIXEL_ARRAY *, PIXEL_ARRAY *, int );

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    printf("Usage: ./ <image-filename>\n.");
    exit(1);
  }

  int rank, numTasks;
  int rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS)
  {
    printf("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numTasks);

  if(rank == MASTER){
    printf("Number of tasks %i\n", numTasks);
  }

  PIXEL_ARRAY pixel_array;
  PIXEL_ARRAY hybrid_filtered_pixel_array;

  Image charImg;

  if (!ImageLoad(argv[1], &charImg))
  {
    printf("ERROR loading image.");
    return;
  }
  createPixelArrayFromImage(charImg, &pixel_array);

  createPixelArrayFromImage(charImg, &hybrid_filtered_pixel_array);

  int imgSliceSize = (pixel_array.sizeY / numTasks) * pixel_array.sizeX;

  /* Timing */

  double time1 = omp_get_wtime();
  mpiDistribute(&hybrid_filtered_pixel_array, &pixel_array, WINDOW_SIZE, imgSliceSize, rank, numTasks);
  double timeomp = omp_get_wtime() - time1;

  if (rank == MASTER)
  {
    PIXEL_ARRAY naive_filtered_pixel_array;

    createPixelArrayFromImage(charImg, &naive_filtered_pixel_array);
    /* Time and call serial version */
    /* Warmup */
    averagingFilterNaive(&naive_filtered_pixel_array, &pixel_array, WINDOW_SIZE);

    /* Timing */
    time1 = omp_get_wtime();
    averagingFilterNaive(&naive_filtered_pixel_array, &pixel_array, WINDOW_SIZE);
    double timeserial = omp_get_wtime() - time1;

    printf("Elapsed time, serial filtering (s) = %f\n", timeserial);
    /*Checks that the two pictures are identical*/
    if (0 == memcmp(hybrid_filtered_pixel_array.data,
                    naive_filtered_pixel_array.data,
                    naive_filtered_pixel_array.sizeX * naive_filtered_pixel_array.sizeY * sizeof(int32_t)))
    {
      printf("Elapsed time, using omp filtering (s) = %f\n", timeomp);
    }
    else
    {
      printf("ERROR creating omp image, image not identical to serial filtered version\n");
    }
  }
}


void mpiDistribute(PIXEL_ARRAY *img, PIXEL_ARRAY *orig_pixel_array, int N, int sliceSize, int rank, int numRanks)
{
  int radius = N / 2;
  //  Scatter pictureslices among mpi tasks

  int halosize = radius*2*orig_pixel_array->sizeX;
  //recievebuffer
  int32_t *pixel_data_slice = malloc((sliceSize  + halosize)* sizeof(int32_t));
  
  MPI_Scatter(
      orig_pixel_array->data,
      sliceSize,
      MPI_INT32_T,
      pixel_data_slice + (halosize* sizeof(int32_t)),
      sliceSize,
      MPI_INT32_T,
      MASTER,
      MPI_COMM_WORLD
      );

// send bottom of slice to bottom neighbour and recive top of bottom neighbours slice
if(numRanks-1 > rank ){
  MPI_Sendrecv( 
    pixel_data_slice + sliceSize + (halosize/2), 
    halosize/2, 
    MPI_INT32_T,
    rank +1, 
    1, //tag
    pixel_data_slice, 
    halosize/2, 
    MPI_INT32_T,
    rank -1, 
    1, //tag
    MPI_COMM_WORLD, 
    NULL //status
    );
  }
  // send top of slice to top neighbour and recieve bottom of top neighbours slice
  if( rank > 0){
  MPI_Sendrecv(
    pixel_data_slice,
    halosize/2, 
    MPI_INT32_T,
    rank -1, 
    1, //tag
    pixel_data_slice + sliceSize + (halosize/2),  
    halosize/2, 
    MPI_INT32_T,
    rank +1, 
    1, //tag
    MPI_COMM_WORLD, 
    NULL //status
    );
  }


  
  PIXEL_ARRAY haloedPixels = {.sizeX =(*orig_pixel_array).sizeX, .sizeY = sliceSize/ (*orig_pixel_array).sizeX , .data = pixel_data_slice};
  
  PIXEL_ARRAY *filteredHaloedPixels;
printf("sizeof frist %i", sizeof(haloedPixels));
  memcpy(filteredHaloedPixels, &haloedPixels, sizeof(PIXEL_ARRAY));
  //filteredHaloedPixels->data = malloc(sizeof(pixel_data_slice));

  // averagingFilterOpenMP(filteredHaloedPixels, &haloedPixels, N);
  // int32_t *returnData = malloc(sliceSize* sizeof(int32_t));
  // memcpy(returnData, filteredHaloedPixels+(halosize/2), sizeof(sliceSize));
  // MPI_Gather ( 
  //   returnData, 
  //   sliceSize* sizeof(int32_t), 
  //   MPI_INT32_T,
  //   img->data, 
  //   sizeof(*(img->data)), 
  //   MPI_INT32_T,
  //   MASTER, 
  //   MPI_COMM_WORLD );
  // combine pictureslices and halo into one array
  //openmp parallel for on the array
  //do calculation and return noisereduced array
  // mpi gather noicereduced arrays -- make it blocking for synchonizing time measurement
  MPI_Finalize();

}

void averagingFilterOpenMP(PIXEL_ARRAY *img, PIXEL_ARRAY *orig_img, int N)
{
  /* Fill Me In! */
  int radius = N / 2;

#pragma omp parallel for \
default(none) shared(img, radius, orig_img, N) \
schedule(dynamic)

  for (int i = 0; i < img->sizeY; i++)
  {

    for (int j = 0; j < (img->sizeX); j++)
    {
      /* For pixels whose window would extend out of bounds, we need to count
    the amount of pixels that we miss, since the window size will be smaller */
      int out_of_bounds = 0;

      /* We are going to average the rgb values over the window */
      int red_avg = 0;
      int blue_avg = 0;
      int green_avg = 0;

      /* This for loop sums up the rgb values for each pixel in the window */
      for (int n = i - radius; n <= i + radius; n++)
      {
        for (int m = j - radius; m <= j + radius; m++)
        {
          /*  If we have an edge pixel, some of the window pixels will
      be out of bounds. Thus we skip these and note that the
      amount of pixels in the window are less than the window size */
          if (n < 0 || m < 0 || n >= img->sizeY || m >= img->sizeX)
          {
            out_of_bounds++;
            continue;
          }
          int idx = m + n * img->sizeX;
          /* Shift, mask and add */
          red_avg += ((orig_img->data[idx] >> 16) & 0xFF);
          green_avg += ((orig_img->data[idx] >> 8) & 0xFF);
          blue_avg += (orig_img->data[idx] & 0xFF);
        }
      }

      /* Divide the total sum by the amount of pixels in the window */
      red_avg /= (N * N - out_of_bounds);
      green_avg /= (N * N - out_of_bounds);
      blue_avg /= (N * N - out_of_bounds);

      /* Set the average to the current pixel */
      int curr_idx = j + i * img->sizeX;
      int32_t pixel = (red_avg << 16) + (green_avg << 8) + blue_avg;
      img->data[curr_idx] = pixel;
    }
  }
}


//copy-pasted from filter.c
void averagingFilterNaive(PIXEL_ARRAY *img, PIXEL_ARRAY *orig_img, int N)
{

  if (N % 2 == 0)
  {
    printf("ERROR: Please use an odd sized window\n");
    exit(1);
  }

  int radius = N / 2;

  for (int i = 0; i < img->sizeY; i++)
  {
    for (int j = 0; j < img->sizeX; j++)
    {
      /* For pixels whose window would extend out of bounds, we need to count
    the amount of pixels that we miss, since the window size will be smaller */
      int out_of_bounds = 0;

      /* We are going to average the rgb values over the window */
      int red_avg = 0;
      int blue_avg = 0;
      int green_avg = 0;

      /* This for loop sums up the rgb values for each pixel in the window */
      for (int n = i - radius; n <= i + radius; n++)
      {
        for (int m = j - radius; m <= j + radius; m++)
        {
          /*  If we have an edge pixel, some of the window pixels will
      be out of bounds. Thus we skip these and note that the
      amount of pixels in the window are less than the window size */
          if (n < 0 || m < 0 || n >= img->sizeY || m >= img->sizeX)
          {
            out_of_bounds++;
            continue;
          }
          int idx = m + n * img->sizeX;
          /* Shift, mask and add */
          red_avg += ((orig_img->data[idx] >> 16) & 0xFF);
          green_avg += ((orig_img->data[idx] >> 8) & 0xFF);
          blue_avg += (orig_img->data[idx] & 0xFF);
        }
      }
      /* Divide the total sum by the amount of pixels in the window */
      red_avg /= (N * N - out_of_bounds);
      green_avg /= (N * N - out_of_bounds);
      blue_avg /= (N * N - out_of_bounds);

      /* Set the average to the current pixel */
      int curr_idx = j + i * img->sizeX;
      int32_t pixel = (red_avg << 16) + (green_avg << 8) + blue_avg;
      img->data[curr_idx] = pixel;
    }
  }
}
