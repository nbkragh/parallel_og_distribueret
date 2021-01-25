/* Exercise: Pi                                                                 
 *                                                                              
 * In this exercise you will determine the value                                
 * of PI using the integral  of                                                 
 *    4/(1+x*x) between 0 and 1.                                                
 *                                                                              
 * The integral is approximated by a sum of n intervals.                        
 *                                                                              
 * The approximation to the integral in each interval is:                       
 *    (1/n)*4/(1+x*x).                                                          
 */
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#define PI25DT 3.141592653589793238462643

#define INTERVALS 10000000
//global variabels
int number_of_threads, intervals = INTERVALS;
double dx, global_sum;
void *calcSum(void *rank);
double timeSpend(struct timespec finish, struct timespec start);
pthread_mutex_t mutex;

int main(int argc, char **argv)
{
  double time, shortestTime = INTERVALS, firstTime = 1, pi;
  int repeat = 32, i;
  struct timespec start, finish;

  struct timeMeasure
  {
    double time;
    double pi;
  } measurements[16];

  struct timespec res;
  clock_getres(CLOCK_MONOTONIC, &res);
  printf("resolution: %.10f\n", res.tv_nsec / (double)1000000000);
  long thread_rank; // is converted to void, which is long on 64bit system
  //number_of_threads = strtol(argv[1], NULL, 10);

  for (number_of_threads = 1; number_of_threads <= 16; number_of_threads++)
  {

    pthread_t *thread_handles = malloc(number_of_threads * sizeof(pthread_t));

    pthread_mutex_init(&mutex, NULL);

    dx = 1.0 / (double)intervals;
    global_sum = 0.0;
    for (i = 0; i < repeat; i++)
    {
      clock_gettime(CLOCK_MONOTONIC, &start);
      for (thread_rank = 0; thread_rank < number_of_threads; thread_rank++)
      {
        pthread_create(&thread_handles[thread_rank], NULL, calcSum, (void *)thread_rank);
      }

      for (thread_rank = 0; thread_rank < number_of_threads; thread_rank++)
      {
        pthread_join(thread_handles[thread_rank], NULL);
      }

      clock_gettime(CLOCK_MONOTONIC, &finish);
      time = timeSpend(finish, start);
      if (shortestTime > time)
      {
        shortestTime = time;
      }
      if (number_of_threads == 1)
      {
        firstTime = shortestTime;
      }
    }
    free(thread_handles);
    pthread_mutex_destroy(&mutex); //destroying mutex as the book tells me too

    pi = dx * (global_sum / repeat);
    measurements[number_of_threads-1].pi = pi;
    measurements[number_of_threads-1].time = shortestTime;
  }
  for (i = 0; i < 16; i++)
  {
    printf("-------------threads %d ----------------\n", i + 1);
    printf("Computed PI %.24f\n", measurements[i].pi);
    printf("The true PI %.24f\n", PI25DT);
    printf("Error       %.24f\n\n", PI25DT - measurements[i].pi);
    printf("parallel time (s) = %f\n", measurements[i].time);
    printf("________________________________________\n");
  }

  for (i = 0; i < 16; i++)
  {
    printf("%f,\n", measurements[i].time);
  }

  printf(" time / speedup  / efficiency \n");
  for (i = 0; i < 16; i++)
  {

    printf("%.2f & %.2f & %.2f\n", measurements[i].time*100, firstTime / measurements[i].time, (firstTime / measurements[i].time) / (i + 1));
  }
  return 0;
}

void *calcSum(void *rank)
{
  long local_rank = (long)rank; //local variabel with rank number
  double local_sum = 0.0, f, x;
  int j;
  for (j = (intervals / number_of_threads) * (local_rank + 1); j >= (intervals / number_of_threads) * local_rank; j--)
  {
    x = dx * ((double)(j - 0.5));
    f = 4.0 / (1.0 + x * x);
    local_sum += f;
  }
  //critical section
  pthread_mutex_lock(&mutex);
  global_sum += local_sum;
  pthread_mutex_unlock(&mutex);
  // end critical section
  return NULL;
}

double timeSpend(struct timespec finish, struct timespec start)
{
  long seconds = finish.tv_sec - start.tv_sec;
  long nanoSeconds = finish.tv_nsec - start.tv_nsec;

  if (start.tv_nsec > finish.tv_nsec)
  { // correcting negative nanoseconds
    --seconds;
    nanoSeconds += 1000000000;
  }

  return (double)seconds + ((double)nanoSeconds / (double)1000000000);
}
