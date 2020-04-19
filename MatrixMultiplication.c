#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>


#define N 500

int X[N][N];
int Y[N][N];
int Z[N][N];

int main() 
{
    int i,j,k;
    struct timeval t1, t2;
    struct timezone tz;
	double time_taken; 
    omp_set_num_threads(omp_get_num_procs());
	 
	 //Matrix creation with random values.
    for (i= 0; i< N; i++)
        for (j= 0; j< N; j++)
	{
            X[i][j] = rand();
            Y[i][j] = rand();
	}
	gettimeofday(&t1, &tz); //Current time stamp
	
	
	
	
	//Simple Matrix multiplication without using parallelism
	
    for (i = 0; i < N; ++i) {
		
        for (j = 0; j < N; ++j) {
			
            for (k = 0; k < N; ++k) {
				
                Z[i][j] += X[i][k] * Y[k][j];
            }
        }
    }
	
	
    gettimeofday(&t2, &tz);
    time_taken = (double) (t2.tv_sec-t1.tv_sec) + (double) (t2.tv_usec-t1.tv_usec) * 1.e-6;
    printf("Time taken using Sequential Matrix = %f seconds.\n", time_taken);
	
	
	gettimeofday(&t1, &tz); //Current time stamp
	
	//Using #pragma OpenMP Parallel with private variable i,j,k
	
    #pragma omp parallel for private(i,j,k) shared(X,Y,Z)
	
    for (i = 0; i < N; ++i) {
		
        for (j = 0; j < N; ++j) {
			
            for (k = 0; k < N; ++k) {
				
                Z[i][j] += X[i][k] * Y[k][j];
            }
        }
    }
	
	
    gettimeofday(&t2, &tz);
    time_taken = (double) (t2.tv_sec-t1.tv_sec) + (double) (t2.tv_usec-t1.tv_usec) * 1.e-6;
    printf("Time taken using Pragma omp = %f seconds.\n", time_taken);

  
}