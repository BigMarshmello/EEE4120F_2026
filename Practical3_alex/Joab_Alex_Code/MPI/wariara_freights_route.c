// =========================================================================
// Practical 3: Minimum Energy Consumption Freight Route Optimization
// =========================================================================
//
// GROUP NUMBER:
//
// MEMBERS:
//   - Alex Hillman, HLLALE010
//   - Joab Kloppers, KLPJOA002

// ========================================================================
//  PART 2: Minimum Energy Consumption Freight Route Optimization using OpenMPI
// =========================================================================


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <string.h>
#include "mpi.h"

#define MAX_N 10

// ============================================================================
// Global variables
// ============================================================================

int n; // If this is -1, it signals an error/exit
int adj[MAX_N][MAX_N];
int best_cost;
int best_path[MAX_N];
int path[MAX_N];
int visited[MAX_N];
double t_init_start, t_init_end, t_comp_start, t_comp_end;

// ============================================================================
// Timer: returns time in seconds
// ============================================================================

double gettime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// ============================================================================
// Usage function
// ============================================================================

void Usage(char *program) {
  printf("Usage: mpirun -np <num> %s [options]\n", program);
  printf("-i <file>\tInput file name\n");
  printf("-o <file>\tOutput file name\n");
  printf("-h \t\tDisplay this help\n");
}



void search(int level, int current_cost) {
    if (current_cost >= best_cost) return;

    if (level == n) {
        best_cost = current_cost;
        for (int k = 0; k < n; k++) {
            best_path[k] = path[k];
        }
        return;
    }

    int last = path[level - 1];

    for (int next = 1; next < n; next++) {
        if (!visited[next]) {
            visited[next] = 1;
            path[level] = next;

            search(level + 1, current_cost + adj[last][next]);

            visited[next] = 0;
        }
    }
}


int main(int argc, char **argv)
{
    int rank, nprocs;
    int opt;
    int i, j;
    char *input_file = NULL;
    char *output_file = NULL;
    FILE *infile = NULL;
    FILE *outfile = NULL;
    int success_flag = 1; // 1 = good, 0 = error/help encountered

    // Initialize MPI
    MPI_Init(&argc, &argv);
    t_init_start = gettime();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


    if (rank == 0) {
        n = -1; 

        while ((opt = getopt(argc, argv, "i:o:h")) != -1)
        {
            switch (opt)
            {
                case 'i':
                    input_file = optarg;
                    break;

                case 'o':
                    output_file = optarg;
                    break;

                case 'h':
                    Usage(argv[0]);
                    success_flag = 0; 
                    break;

                default:
                    Usage(argv[0]);
                    success_flag = 0;
            }
        }

        
    
        if (success_flag) {
            infile = fopen(input_file, "r");
            if (infile == NULL) {
                fprintf(stderr, "Error: Cannot open input file '%s'\n", input_file);
                perror("");
                success_flag = 0;
            } else {
                
                fscanf(infile, "%d", &n);

                for (i = 1; i < n; i++)
                {
                    for (j = 0; j < i; j++)
                    {
                        fscanf(infile, "%d", &adj[i][j]);
                        adj[j][i] = adj[i][j];
                    }
                }
                fclose(infile);
            }
        }
        if (success_flag) {
            outfile = fopen(output_file, "w");
            if (outfile == NULL) {
                fprintf(stderr, "Error: Cannot open output file '%s'\n", output_file);
                perror("");
                success_flag = 0;
            }
        }

    }


    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    
    if (n == -1) {
        MPI_Finalize();
        return 0; 
    }

   

  
    MPI_Bcast(&adj[0][0], MAX_N * MAX_N, MPI_INT, 0, MPI_COMM_WORLD);

    /*
    printf("Process %d received adjacency matrix:\n", rank);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%d ", adj[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    */

        
    // TODO: compute solution to minimum energy consumption problem here and write to output file
    // Be careful on which process rank writes to the output file to avoid conflicts!
       // printf("Process %d reached solve section\n", rank);
        t_init_end = gettime();
        t_comp_start = gettime();
        int local_best_cost = 999999;
        int local_best_path[MAX_N];

        for (int second = 1 + rank; second < n; second += nprocs) {
            for (i = 0; i < MAX_N; i++) {
                visited[i] = 0;
            }

            best_cost = 999999;
            path[0] = 0;
            path[1] = second;
            visited[0] = 1;
            visited[second] = 1;

            search(2, adj[0][second]);

            if (best_cost < local_best_cost) {
                local_best_cost = best_cost;
                for (i = 0; i < n; i++) {
                    local_best_path[i] = best_path[i];
                }
            }
        }

        best_cost = local_best_cost;
        for (i = 0; i < n; i++) {
            best_path[i] = local_best_path[i];
        }
        int global_best_cost;
        int winner_rank = -1;
        MPI_Reduce(&local_best_cost, &global_best_cost, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_best_cost, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (local_best_cost == global_best_cost) {
            winner_rank = rank;
        }
        int global_winner_rank;
        MPI_Reduce(&winner_rank, &global_winner_rank, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_winner_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == global_winner_rank && rank != 0) {
            MPI_Send(local_best_path, n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }       

        if (rank == 0 && global_winner_rank != 0) {
            MPI_Recv(best_path, n, MPI_INT, global_winner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == 0 && global_winner_rank == 0) {
            for (i = 0; i < n; i++) {
            best_path[i] = local_best_path[i];
            }
        }
        t_comp_end = gettime();
        if (rank == 0) {

        best_cost = global_best_cost;
        //best_cost = 999999;
            /*
        for (i = 0; i < MAX_N; i++) {
            visited[i] = 0;
        }

        path[0] = 0;
        visited[0] = 1;
        */
        //search(1, 0);
       // printf("Winner rank: %d\n", global_winner_rank);
        //printf("Best path: ");
        //for (i = 0; i < n; i++) {
        //    printf("%d ", best_path[i] + 1);
        //}
        //printf("\nBest cost: %d\n", best_cost);
        printf("Tinit:  %.6f seconds\n", t_init_end - t_init_start);
        printf("Tcomp:  %.6f seconds\n", t_comp_end - t_comp_start);


        fprintf(outfile, "Best path: ");
        for (i = 0; i < n; i++) {
            fprintf(outfile, "%d ", best_path[i] + 1);
        }
        fprintf(outfile, "\nBest cost: %d\n", best_cost);
        fprintf(outfile, "Tinit: %.6f seconds\n", t_init_end - t_init_start);
        fprintf(outfile, "Tcomp: %.6f seconds\n", t_comp_end - t_comp_start);
        fclose(outfile);
    }

    

    MPI_Finalize();
    return 0;
}