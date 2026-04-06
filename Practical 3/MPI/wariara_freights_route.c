// =========================================================================
// Practical 3: Minimum Energy Consumption Freight Route Optimization
// =========================================================================
//
// GROUP NUMBER:
//
// MEMBERS:
//   - Member 1 Name, Student Number
//   - Member 2 Name, Student Number

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

typedef struct {
    int    path[MAX_N];
    int    path_len;
    int    visited[MAX_N];
    int    cost;
} Node;
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

// ============================================================================
// Sequential branch and bound (no OpenMP — each MPI rank runs this alone)
// ============================================================================
 
void branch_and_bound(Node node, int depth)
{
    if (node.cost >= best_cost) return;
 
    if (node.path_len == n)
    {
        int start = node.path[0];
        int last  = node.path[node.path_len - 1];
        int total = node.cost + adj[last][start];
 
        if (total < best_cost)
        {
            best_cost = total;
            memcpy(best_path, node.path, n * sizeof(int));
        }
        return;
    }
 
    int last_city = node.path[node.path_len - 1];
 
    for (int next = 0; next < n; next++)
    {
        if (node.visited[next]) continue;
        if (adj[last_city][next] >= 9999) continue;
 
        Node child                   = node;
        child.path[child.path_len++] = next;
        child.visited[next]          = 1;
        child.cost                  += adj[last_city][next];
 
        if (child.cost >= best_cost) continue;
 
        branch_and_bound(child, depth + 1);
    }
}
 
// ============================================================================
// Work generation: expand the root to a pool of partial paths of fixed depth,
// so each MPI rank gets a roughly equal share to explore.
// ============================================================================
 
#define MAX_WORK_ITEMS 1000
 
Node work_pool[MAX_WORK_ITEMS];
int  work_count = 0;
 
void generate_work(Node node, int target_depth)
{
    if (node.path_len == target_depth || node.path_len == n)
    {
        if (work_count < MAX_WORK_ITEMS)
            work_pool[work_count++] = node;
        return;
    }
 
    int last_city = node.path[node.path_len - 1];
 
    for (int next = 0; next < n; next++)
    {
        if (node.visited[next]) continue;
        if (adj[last_city][next] >= 9999) continue;
 
        Node child                   = node;
        child.path[child.path_len++] = next;
        child.visited[next]          = 1;
        child.cost                  += adj[last_city][next];
 
        generate_work(child, target_depth);
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

    double t_init_start = gettime();

    // Initialize MPI
    MPI_Init(&argc, &argv);
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

   

    
    //MPI_Bcast(&adj[0][0], MAX_N * MAX_N, MPI_INT, 0, MPI_COMM_WORLD);

    
    //printf("Process %d received adjacency matrix:\n", rank);
    //for (i = 0; i < n; i++) {
    //    for (j = 0; j < n; j++) {
    //        printf("%d ", adj[i][j]);
    //    }
    //    printf("\n");
    //}
    //printf("\n");

        
    // TODO: compute solution to minimum energy consumption problem here and write to output file
    // Be careful on which process rank writes to the output file to avoid conflicts!
    
    double t_av        = 0.0;
    double t_init_end  = gettime();
 
    // Choose how deep to expand the work tree before distributing.
    // Depth 3 gives up to n*(n-1)*(n-2) partial paths — enough for many ranks.
    int work_depth = 3;
 
    //for (int iter = 0; iter < 1000; iter++)
    //{
        // Reset local best for this iteration
        best_cost = 9999;
        memset(best_path, 0, sizeof(best_path));
 
        // ------------------------------------------------------------------
        // Rank 0 generates the work pool and scatters it.
        // All other ranks wait to receive their slice.
        // ------------------------------------------------------------------
        int my_work_count = 0;
        Node my_work[MAX_WORK_ITEMS];
 
        if (rank == 0)
        {
            work_count = 0;
 
            Node root       = {0};
            root.path[0]    = 0;
            root.path_len   = 1;
            root.visited[0] = 1;
            root.cost       = 0;
 
            generate_work(root, work_depth);
        }
 
        // Broadcast total work count so every rank knows loop bounds
        MPI_Bcast(&work_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
 
        // Broadcast the full work pool — all ranks need to index into it
        MPI_Bcast(work_pool, work_count * sizeof(Node), MPI_BYTE, 0, MPI_COMM_WORLD);
 
        // ------------------------------------------------------------------
        // Static work distribution: each rank takes every (size)-th item.
        // This round-robin assignment keeps load balanced without extra comms.
        // ------------------------------------------------------------------
        MPI_Barrier(MPI_COMM_WORLD); // Synchronise before timing
        double t_start = gettime();
 
        for (int w = rank; w < work_count; w += nprocs)
        {
            my_work[my_work_count++] = work_pool[w];
        }
 
        for (int w = 0; w < my_work_count; w++)
        {
            branch_and_bound(my_work[w], work_depth);
        }
 
        // ------------------------------------------------------------------
        // Reduction: find the rank that achieved the global minimum cost,
        // then broadcast that rank's best_path to all ranks.
        // ------------------------------------------------------------------
 
        // MPI_Reduce with MPI_MINLOC finds the minimum cost and which rank holds it
        struct { int cost; int rank; } local_result, global_result;
        local_result.cost = best_cost;
        local_result.rank = rank;
 
        MPI_Reduce(&local_result, &global_result, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
 
        // The winning rank broadcasts its best_path to all (rank 0 coordinates)
        int winning_rank = 0;
        if (rank == 0)
        {
            winning_rank = global_result.rank;
            best_cost    = global_result.cost;     // rank 0 stores global best
        }
        MPI_Bcast(&winning_rank, 1,           MPI_INT,  0,            MPI_COMM_WORLD);
        MPI_Bcast(&best_cost,    1,           MPI_INT,  0,            MPI_COMM_WORLD);
        MPI_Bcast(best_path,     MAX_N,       MPI_INT,  winning_rank, MPI_COMM_WORLD);
 
        MPI_Barrier(MPI_COMM_WORLD); // Synchronise after timing
        double t_end = gettime();
 
        t_av += t_end - t_start;
    //}
 
    // -------------------------------------------------------------------------
    // Output — only rank 0 prints/writes results (mirrors OpenMP single-writer)
    // -------------------------------------------------------------------------
    if (rank == 0)
    {
        double t_compute = t_av;// / 1000.0;
        double t_init    = t_init_end - t_init_start;
 
        printf("Tinit:  %.6f seconds\n", t_init);
        printf("Tcomp:  %.6f seconds\n", t_compute);
 
        // Write best path and cost to output file
        fprintf(outfile, "Best cost: %d\n", best_cost);
        fprintf(outfile, "Best path: ");
        for (int k = 0; k < n; k++)
            fprintf(outfile, "%d ", best_path[k]);
        fprintf(outfile, "\n");
 
        fclose(outfile);
    }

    MPI_Finalize();
    return 0;
}