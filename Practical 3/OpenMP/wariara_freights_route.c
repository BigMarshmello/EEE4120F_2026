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
#include <math.h>

#define MAX_N 10

// ============================================================================
// Global variables
// ============================================================================

int procs = 1;

int n;
int adj[MAX_N][MAX_N];


int best_cost;
omp_lock_t global_update_lock;
int best_path[MAX_N];

typedef struct {
    int    path[MAX_N];
    int    path_len;
    int    visited[MAX_N];
    double cost;               // only field we prune on now
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
  printf("Usage: %s [options]\n", program);
  printf("-p <num>\tNumber of processors/threads to use\n");
  printf("-i <file>\tInput file name\n");
  printf("-o <file>\tOutput file name\n");
  printf("-h \t\tDisplay this help\n");
}

// ============================================================================
// branch and bound algorithm
// ============================================================================

void branch_and_bound(Node node, int depth)
{
    //double time_start = gettime();

    int current_best;
    #pragma omp atomic read
    current_best = best_cost;

    if (node.cost >= current_best) return;

    if (node.path_len == n) 
    {
        int start    = node.path[0];
        int last     = node.path[node.path_len - 1];
        int total = node.cost + adj[last][start];

        if (total < current_best) 
        {
            //#pragma omp critical
            omp_set_lock(&global_update_lock);
            if (total < best_cost) 
            {
                
                best_cost = total;

                memcpy(best_path, node.path, n * sizeof(int));
                
                //printf("[Thread %d] New best: %i\n",omp_get_thread_num(), best_cost);
            
            }
            omp_unset_lock(&global_update_lock);
        }
        return;
    }

        int last_city = node.path[node.path_len - 1];

    for (int next = 0; next < n; next++) {

        if (node.visited[next]) continue;
        if (adj[last_city][next] >= 9999) continue;

        Node child             = node;
        child.path[child.path_len++] = next;
        child.visited[next]    = 1;
        child.cost            += adj[last_city][next];

        // Pre-prune before spawning a task
        int best_now;
        #pragma omp atomic read
        best_now = best_cost;

        if (child.cost >= best_now) continue;

        if (depth < 4) {
            #pragma omp task firstprivate(child)
            branch_and_bound(child, depth + 1);
        } else {
            branch_and_bound(child, depth + 1);
        }
    }

    #pragma omp taskwait
    
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

 
    omp_set_num_threads(procs);
    printf("Running with %d processes/threads on a graph with %d nodes\n", procs, n);

    
    // TODO: compute solution to minimum energy consumption problem here and write to outfile
    
    double t_av = 0;
    double t_init_end = gettime();
    omp_init_lock(&global_update_lock);
  

    for (int iter = 0; iter < 1000; iter++)
    {
        //printf("iteration %i\n",iter);
        Node root       = {0};
        root.path[0]    = 0;
        root.path_len   = 1;
        root.visited[0] = 1;
        root.cost       = 0.0;
        best_cost     = 9999;
        
        double t_start = gettime();
        #pragma omp parallel
        #pragma omp single
        branch_and_bound(root, 0);

        double t_end = gettime();

        omp_destroy_lock(&global_update_lock);

        t_av += t_end-t_start;
    }

    double t_compute = t_av/1000;
    double t_init = t_init_end-t_init_start;

    printf("Tinit:  %.6f seconds\n", t_init);
    printf("Tcomp:  %.6f seconds\n", t_compute);
    //printf("Ttotal: %.6f seconds\n", t_init + t_comp);

    fclose(outfile);
    

    return 0;
}
