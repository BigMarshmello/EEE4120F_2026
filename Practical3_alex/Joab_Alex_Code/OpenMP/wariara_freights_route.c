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
//  PART 1: Minimum Energy Consumption Freight Route Optimization using OpenMP
// =========================================================================


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
int n; // number of cities
int adj[MAX_N][MAX_N]; //symmetric energy cost matrix

int best_cost;
int best_path[MAX_N];
int path[MAX_N];
int visited[MAX_N];
double t_init_start, t_init_end, t_comp_start, t_comp_end;
int global_best_cost;
int global_best_path[MAX_N];
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

/*void search(int level, int current_cost) {
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
}*/

/* Recursive Brand-and-Bound search
   Each thread passes its own local path, visited array and local best result
   This avoids race conditions during recursive exploration
*/
void search(int level, int current_cost, int local_path[], int local_visited[],
            int *local_best_cost, int local_best_path[]) {

    // Prune branch if partial route is already worse than current local best
    if (current_cost >= *local_best_cost) return; 

    // If a full route has been built, update the thread's local best solution
    if (level == n) {
        *local_best_cost = current_cost;
        for (int k = 0; k < n; k++) {
            local_best_path[k] = local_path[k];
        }
        return;
    }

    int last = local_path[level - 1];

    // Try all unvisited next cities, recurse then backtrack
    for (int next = 1; next < n; next++) {
        if (!local_visited[next]) {
            local_visited[next] = 1;
            local_path[level] = next;

            search(level + 1,
                   current_cost + adj[last][next],
                   local_path,
                   local_visited,
                   local_best_cost,
                   local_best_path);

            local_visited[next] = 0;
        }
    }
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
    
    // the start of initialisation time before argument parsing and file setup
    t_init_start = gettime();

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
    // Set the number of OpenMP threads requested by the user
    omp_set_num_threads(procs);
    printf("OpenMP max threads set to: %d\n", omp_get_max_threads());
    printf("Running with %d processes/threads on a graph with %d nodes\n", procs, n);

    
    // TODO: compute solution to minimum energy consumption problem here and write to outfile

    /*printf("Adjacency matrix:\n");
      for (i = 0; i < n; i++) {
          for (j = 0; j < n; j++) {
            printf("%4d ", adj[i][j]);
        }
        printf("\n");
    }
    */
   /* best_cost = 999999;

    for (i = 0; i < MAX_N; i++) {
        visited[i] = 0;
    }

    path[0] = 0;
    visited[0] = 1;
    t_init_end = gettime();
    t_comp_start = gettime();
    search(1, 0);
    t_comp_end = gettime();*/
    
    // Initialise shared best cost before parallel search
    global_best_cost = 999999; //start as large value
    t_init_end = gettime(); //end initialisation timing before computation timing

    t_comp_start = gettime(); //start computation timing

    // Parallelise the first level of the recursion tree
    // Each thread explores one or more second-city branches independently
    /*
    Parallelised first leve of search tree -> route always starts at city 1
    internally as index 0 -> next decision is second city and each second
    city choice becomes one subtree. Threads take different values of second. 

    Dynamic scheduling helps balance irregular Branch-and-Bound workloads
    */
    #pragma omp parallel for private(i) schedule(dynamic)
    for (int second = 1; second < n; second++) {
        //printf("Thread %d handling second city %d\n", omp_get_thread_num(), second + 1);

        // each thread has its own local:
        int local_visited[MAX_N] = {0};
        int local_path[MAX_N];
        int local_best_cost = 999999;
        int local_best_path[MAX_N];
        for (i = 0; i < MAX_N; i++) {
            local_visited[i] = 0;
        }

        local_best_cost = 999999;

        // Fix the first two cities of the route: start at city 1 (index 0), then assign this thread's second-city branch
        local_path[0] = 0;
        local_path[1] = second;
        local_visited[0] = 1;
        local_visited[second] = 1;

       /* for (i = 0; i < MAX_N; i++) {
            visited[i] = 0;
        }

        best_cost = 999999;

        path[0] = 0;
        path[1] = second;
        visited[0] = 1;
        visited[second] = 1;*/

        //search(2, adj[0][second]);

        //recursively search the remainder of the subtree
        search(2, adj[0][second], local_path, local_visited, &local_best_cost, local_best_path);

        // safely merge the threads local best reuslt into the shared global
        // critical prevents simultaneous updates by multiple threads
        #pragma omp critical
        {
            if (local_best_cost < global_best_cost) {
                global_best_cost = local_best_cost;
                for (i = 0; i < n; i++) {
                    global_best_path[i] = local_best_path[i];
                }
            }
        }
    }

    // end computation time
    t_comp_end = gettime();

    best_cost = global_best_cost;
    for (i = 0; i < n; i++) {
        best_path[i] = global_best_path[i];
    }

    //printf("Best path: ");
    //for (i = 0; i < n; i++) {
     //   printf("%d ", best_path[i]+1);
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

    return 0;
}
