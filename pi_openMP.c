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

#include <stdio.h>
#include <time.h>
#include <omp.h>
#define PI25DT 3.141592653589793238462643

#define INTERVALS 10000000

double calcSum(int dx, long intervals, int nthreads);
double timeSpend(struct timespec finish, struct timespec start);

int main(int argc, char **argv)
{
    int number_of_threads;
    double dx, global_sum = 0.0;
    long int intervals = INTERVALS;
    dx = 1.0 / (double)intervals;
    
    double time, shortestTime = 99999, firstTime = 1, pi;
    int repeat = 32, i;
    struct timespec start, finish;
    struct timeMeasure
    {
        double time;
        double pi;
    } measurements[16];
    for (number_of_threads = 1; number_of_threads <= 16; number_of_threads++)
    {
        for (i = 0; i < repeat; i++)
        {
            clock_gettime(CLOCK_MONOTONIC, &start);

            //critical section
#           pragma omp parallel num_threads(number_of_threads) reduction(+ \
                                                              : global_sum)
            global_sum += calcSum(dx, intervals, number_of_threads);
            // end critical section
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
        pi = (dx * global_sum) / repeat;
        measurements[number_of_threads - 1].pi = pi;
        measurements[number_of_threads - 1].time = shortestTime;
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

        printf("%.2f & %.2f & %.2f\n", measurements[i].time * 100, firstTime / measurements[i].time, (firstTime / measurements[i].time) / (i + 1));
    }

    return 0;
}
double calcSum(int dx, long intervals, int nthreads)
{
    int local_rank = omp_get_thread_num();
    double local_sum = 0.0, f, x;
    int j;
    for (j = (intervals / nthreads) * (local_rank + 1); j >= (intervals / nthreads) * local_rank; j--)
    {
        x = dx * ((double)(j - 0.5));
        f = 4.0 / (1.0 + x * x);
        local_sum += f;
    }

    return local_sum;
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
