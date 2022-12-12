#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

void prt1a(char *t1, double *v, int n,char *t2);

int N;
double *A;
#define A(i,j) A[(i)*(N+1)+(j)]
double *X;

int main(int argc,char **argv) {
    double time0, time1;
    int i, j, k;
    /* create arrays */
    for (N=100; N < 4501; N += 200) {
		A=(double *)malloc(N*(N+1)*sizeof(double));
		X=(double *)malloc(N*sizeof(double));
		printf("\n----------------------------------\nGAUSS %dx%d\n",N,N);
		time0 = omp_get_wtime();
		#pragma omp parallel shared(A,N,X) private(i,j,k)
		{
			#pragma omp for schedule(static) collapse(2)
				/* initialize array A*/
				for(i=0; i <= N-1; i++) {
					for(j=0; j <= N; j++) {
						if (i==j || j==N)
							A(i,j) = 1.f;
						else 
							A(i,j)=0.f;
					}
				}
			/* elimination */
			for (i=0; i<N-1; i++) {
			#pragma omp for schedule(static)
				for (k=i+1; k <= N-1; k++) {
					for (j=i+1; j <= N; j++)
						A(k,j) -= A(k,i)*A(i,j)/A(i,i);
				}
			}
			/* reverse substitution */
			X[N-1] = A(N-1,N)/A(N-1,N-1);
			for (j=N-2; j>=0; j--) {
			#pragma omp for schedule(static)
				for (k=0; k <= j; k++)
					A(k,N) = A(k,N)- A(k,j+1)*X[j+1];
				X[j]=A(j,N)/A(j,j);
			}
		}
		time1 = omp_get_wtime();
		printf("Time in seconds=%gs\n",time1-time0);
		prt1a("X=(", X,N>9?9:N,"...)\n");
		free(A);
		free(X);
    }
    printf("\n");
    return 0;
}

void prt1a(char * t1, double *v, int n,char *t2) {
    int j;
    printf("%s",t1);
    for(j=0;j<n;j++)
        printf("%.4g%s",v[j], j%10==9? "\n": ", ");
    printf("%s",t2);
}
