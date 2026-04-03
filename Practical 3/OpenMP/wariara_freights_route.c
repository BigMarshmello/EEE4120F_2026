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
//  PART 1: Minimum Energy Consumption Freight Route Optimization using OpenMP
// =========================================================================
#include <string.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <omp.h>

#define MAX_N 10

// ============================================================================
// Global variables
// ============================================================================

int procs = 1;

int n;
int adj[MAX_N][MAX_N];


int best_cost;
int best_path[MAX_N];
omp_lock_t best_lock;
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
  printf("Usage: %s [options]\n", program);
  printf("-p <num>\tNumber of processors/threads to use\n");
  printf("-i <file>\tInput file name\n");
  printf("-o <file>\tOutput file name\n");
  printf("-h \t\tDisplay this help\n");
}

//=========================
// Branch and bound recursive code?
//=========================

void branch_and_bound(int path[], int visited[], int depth, int cost) {

    // Prune: no point continuing if already worse than best known
    if (cost >= best_cost) return;

    // Base case: all cities visited, this is a complete route
    if (depth == n) {
        omp_set_lock(&best_lock);
        if (cost < best_cost) {
            best_cost = cost;
            memcpy(best_path, path, n * sizeof(int));
        }
        omp_unset_lock(&best_lock);
        return;
    }

    // Try each unvisited city as the next stop
    for (int next = 0; next < n; next++) {
        if (!visited[next]) {
            int new_cost = cost + adj[path[depth - 1]][next];

            if (new_cost < best_cost) {  // prune early before recursing
                visited[next] = 1;
                path[depth] = next;
                branch_and_bound(path, visited, depth + 1, new_cost);
                visited[next] = 0;  // backtrack
            }
        }
    }
}
//=============================================================================
// Parallel solver — splits top-level branches across threads
// ============================================================================
double solve(FILE *outfile) {

    best_cost = 9999;
    omp_init_lock(&best_lock);
    omp_set_num_threads(procs);

    // Parallelise the first branching level
    // Each iteration tries a different city as the first stop after city 0
    // dynamic scheduling is important — branch sizes vary wildly due to pruning
    double t_start = gettime();
    #pragma omp parallel for schedule(dynamic)
    for (int first = 1; first < n; first++) {

        // Each thread needs its OWN path and visited arrays — never share these
        int local_path[MAX_N];
        int local_visited[MAX_N];
        memset(local_visited, 0, sizeof(local_visited));

        local_path[0] = 0;           // always start at city 0 (City 1 in the prac)
        local_visited[0] = 1;
        local_path[1] = first;
        local_visited[first] = 1;

        branch_and_bound(local_path, local_visited, 2, adj[0][first]);
    }

    double t_end = gettime();
    omp_destroy_lock(&best_lock);


    // Write result to output file (1-indexed as the prac requires)
    fprintf(outfile, "Minimum energy cost: %d\n", best_cost);
    fprintf(outfile, "Route: ");
    for (int i = 0; i < n; i++) {
        fprintf(outfile, "%d ", best_path[i] + 1);
    }
    fprintf(outfile, "\n");

    return t_end-t_start;
}

int main(int argc, char **argv)
{
    
    int opt;
    int i, j;
    char *input_file = NULL;
    char *output_file = NULL;
    FILE *infile = NULL;
    FILE *outfile = NULL;
    int success_flag = 1; // 1 = good, 0 = error/help encountered
    
    double t_init_start = gettime();
    

    while ((opt = getopt(argc, argv, "p:i:o:h")) != -1)
    {
        switch (opt)
        {
            case 'p':
            {
                procs = atoi(optarg);
                break;
            }

            case 'i':
            {
                input_file = optarg;
                break;
            }

            case 'o':
            {
                output_file = optarg;
                break;
            }

            case 'h':
            {
                Usage(argv[0]);
                success_flag = 0; 
                break;
            }

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

    if (!success_flag) return 1;

 

    printf("Running with %d processes/threads on a graph with %d nodes\n", procs, n);

    
    // TODO: compute solution to minimum energy consumption problem here and write to outfile

     // -------------------------------------------------------------------------
    // Computation timing starts here
    // -------------------------------------------------------------------------

    double t_compute = solve(outfile);

    // -------------------------------------------------------------------------

   //printf("Tinit:  %.6f seconds\n", t_init);
    printf("Tcomp:  %.6f seconds\n", t_compute);
    //printf("Ttotal: %.6f seconds\n", t_init + t_comp);

    fclose(outfile);
    

    return 0;
}
