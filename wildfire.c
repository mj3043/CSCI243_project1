/*
 * wildfire.c
 * CSCI 243 – Project 1: Wild Fire Simulation
 * @author Munkh-Orgil Jargalsaikhan
 *
 * A 2-D cellular-automaton that models the spread of fire in a forest.
 *
 * Features
 * --------
 *   • Command-line flags: -H  -bN  -cN  -dN  -nN  -pN  -sN
 *   • Print mode (-pN) – prints N+1 states (including cycle 0)
 *   • Overlay mode – uses display.c for in-place animation
 *   • Fixed random seed 41 → reproducible runs
 *   • No heap allocation for the simulation grid (static 40×40 arrays)
 *   • Burning trees live exactly 3 cycles: * → * → * → .
 *
 * Compile
 * -------
 *   gcc -std=c99 -ggdb -Wall -Wextra -pedantic wildfire.c display.c -lm -o wildfire
 */

 #define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE               /* for usleep() */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "display.h"

/* ============================== SYMBOLIC CONSTANTS ============================== */

/** Possible states of a cell */
enum CellState {
    EMPTY  = 0,   /**< ' ' – empty ground                */
    TREE   = 1,   /**< 'Y' – living tree                */
    BURNING = 2,  /**< '*' – tree on fire               */
    BURNED = 3    /**< '.' – burned-out remains         */
};

/** Grid size limits */
#define MIN_SIZE      5
#define MAX_SIZE     40
#define DEFAULT_SIZE 10

/** Default percentages (command-line overrides) */
#define DEFAULT_PBURN    10   /**< -b : % of trees initially burning   */
#define DEFAULT_PCATCH   30   /**< -c : catch-fire probability         */
#define DEFAULT_DENSITY  50   /**< -d : % of cells that contain trees  */
#define DEFAULT_PNEIGH   25   /**< -n : neighbor-influence threshold   */

/** Fire behaviour */
#define BURN_DURATION    3    /**< cycles a tree stays BURNING          */
#define SIM_DELAY_USEC 750000 /**< overlay-mode pause (0.75 s)          */

/** 8-way neighbour offsets (NW → N → NE → E → SE → S → SW → W) */
static const int dr[] = { -1, -1, -1,  0,  0,  1,  1,  1 };
static const int dc[] = { -1,  0,  1, -1,  1, -1,  0,  1 };

/* ============================== GLOBAL STATE ============================== */

/** The simulation world – static, no heap */
static enum CellState grid[MAX_SIZE][MAX_SIZE];

/** Age of burning cells: -1 = not burning, 0..2 = burning */
static int burn_age[MAX_SIZE][MAX_SIZE];

/** Simulation parameters */
static int sim_size       = DEFAULT_SIZE;
static int pBurning_pct   = DEFAULT_PBURN;
static int pCatch_pct     = DEFAULT_PCATCH;
static int density_pct    = DEFAULT_DENSITY;
static int pNeighbor_pct  = DEFAULT_PNEIGH;
static int print_mode     = 0;   /**< 1 if -pN was supplied */
static int print_limit    = 0;   /**< max cycles in print mode */

/* ============================== FUNCTION PROTOTYPES ============================== */

static void usage(void);
static void error_exit(const char *msg);
static void init_grid(void);
static void display_grid(void);
static void display_stats(int cycle, int changes, int total_changes);
static int  update_simulation(int *changes_out);
static double rand01(void);
static void fisher_yates_shuffle(int *arr, int n);

/* ============================== MAIN ============================== */

/**
 * @brief Program entry point.
 *
 * Parses command-line options, seeds the RNG, builds the initial forest,
 * and runs the simulation in either print or overlay mode.
 *
 * @return EXIT_SUCCESS on normal termination, EXIT_FAILURE on bad input.
 */
int main(int argc, char *argv[]) {
    int opt;
    opterr = 0;                     /* we print our own error messages */

    /* --------------------- command-line processing --------------------- */
    while ((opt = getopt(argc, argv, "Hb:c:d:n:p:s:")) != -1) {
        char *endptr;
        long  val;

        switch (opt) {
            case 'H':
                usage();
                return EXIT_SUCCESS;

            case 'b':
                val = strtol(optarg, &endptr, 10);
                if (*endptr || val < 1 || val > 100)
                    error_exit("(-bN) proportion already burning must be an integer in [1...100].");
                pBurning_pct = (int)val;
                break;

            case 'c':
                val = strtol(optarg, &endptr, 10);
                if (*endptr || val < 1 || val > 100)
                    error_exit("(-cN) probability a tree will catch fire must be an integer in [1...100].");
                pCatch_pct = (int)val;
                break;

            case 'd':
                val = strtol(optarg, &endptr, 10);
                if (*endptr || val < 1 || val > 100)
                    error_exit("(-dN) density of trees in the grid must be an integer in [1...100].");
                density_pct = (int)val;
                break;

            case 'n':
                val = strtol(optarg, &endptr, 10);
                if (*endptr || val < 0 || val > 100)
                    error_exit("(-nN) %neighbors influence catching fire must be an integer in [0...100].");
                pNeighbor_pct = (int)val;
                break;

            case 'p':
                val = strtol(optarg, &endptr, 10);
                if (*endptr || val < 0 || val > 10000)
                    error_exit("(-pN) number of states to print must be an integer in [0...10000].");
                print_mode  = 1;
                print_limit = (int)val;
                break;

            case 's':
                val = strtol(optarg, &endptr, 10);
                if (*endptr || val < MIN_SIZE || val > MAX_SIZE)
                    error_exit("(-sN) simulation grid size must be an integer in [5...40].");
                sim_size = (int)val;
                break;

            case '?':
            default:
                error_exit("Unknown flag.");
        }
    }

    /* --------------------- initialise RNG (fixed seed) --------------------- */
    srandom(41);

    /* --------------------- build the initial forest --------------------- */
    init_grid();

    int cycle          = 0;
    int total_changes  = 0;
    int current_changes = 0;

    /* --------------------- print-mode header --------------------- */
    if (print_mode) {
        printf("===========================\n");
        printf("======== Wildfire =========\n");
        printf("===========================\n");
        printf("=== Print %2d Time Steps ===\n",
               print_limit > 99 ? print_limit : print_limit);
        printf("===========================\n");
    } else {
        clear();                     /* clear terminal once */
    }

    /* --------------------- show cycle 0 --------------------- */
    if (!print_mode) set_cur_pos(1, 0);
    display_grid();
    display_stats(cycle, 0, 0);

    if (print_mode && print_limit == 0) return EXIT_SUCCESS;

    /* --------------------- main simulation loop --------------------- */
    int steps = 0;
    while (1) {
        int still_burning = update_simulation(&current_changes);
        total_changes += current_changes;
        ++cycle;
        ++steps;

        if (!print_mode) set_cur_pos(1, 0);
        display_grid();
        display_stats(cycle, current_changes, total_changes);

        /* termination conditions */
        if (!still_burning) {
            if (!print_mode) {
                usleep(SIM_DELAY_USEC);
                set_cur_pos(sim_size + 4, 0);
            }
            printf("Fires are out.\n");
            break;
        }
        if (print_mode && steps >= print_limit) break;
    }

    return EXIT_SUCCESS;
}

/* ============================== HELP / ERROR ============================== */

/**
 * @brief Print the program usage message to stderr.
 */
static void usage(void) {
    fprintf(stderr,
            "usage: wildfire [options]\n"
            "By default, the simulation runs in overlay display mode.\n"
            "The -pN option makes the simulation run in print mode for up to N states.\n\n"
            "Simulation Configuration Options:\n"
            " -H  # View simulation options and quit.\n"
            " -bN # proportion of trees that are already burning. 0 < N < 101.\n"
            " -cN # probability that a tree will catch fire. 0 < N < 101.\n"
            " -dN # density: the proportion of trees in the grid. 0 < N < 101.\n"
            " -nN # proportion of neighbors that influence a tree catching fire. -1 < N < 101.\n"
            " -pN # number of states to print before quitting. -1 < N < ...\n"
            " -sN # simulation grid size. 4 < N < 41.\n");
}

/**
 * @brief Print an error message, the usage text, and exit with failure.
 * @param msg The specific error text to display.
 */
static void error_exit(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    usage();
    exit(EXIT_FAILURE);
}

/* ============================== RANDOM HELPERS ============================== */

/**
 * @brief Return a pseudo-random double in the interval [0.0, 1.0).
 */
static double rand01(void) {
    return (double)random() / ((double)RAND_MAX + 1.0);
}

/**
 * @brief In-place Fisher-Yates shuffle of an integer array.
 * @param arr Array to shuffle.
 * @param n   Number of elements.
 */
static void fisher_yates_shuffle(int *arr, int n) {
    for (int i = n - 1; i > 0; --i) {
        int j = (int)(rand01() * (i + 1));
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

/* ============================== GRID INITIALISATION ============================== */

/**
 * @brief Populate the grid according to the density and burning percentages.
 *
 * Uses a shuffled linear index array to guarantee uniform random placement.
 */
static void init_grid(void) {
    const int N          = sim_size;
    const int total      = N * N;
    int n_trees   = (int)(density_pct / 100.0 * total + 0.5);
    if (n_trees > total) n_trees = total;

    int n_burning = (int)(pBurning_pct / 100.0 * n_trees + 0.5);
    if (n_burning > n_trees) n_burning = n_trees;

    const int n_live = n_trees - n_burning;

    /* ---- clear everything ---- */
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            grid[r][c]     = EMPTY;
            burn_age[r][c] = -1;
        }

    /* ---- create shuffled index list ---- */
    int *idx = malloc(total * sizeof(int));
    if (!idx) error_exit("Memory allocation failed in init_grid.");
    for (int i = 0; i < total; ++i) idx[i] = i;
    fisher_yates_shuffle(idx, total);

    /* ---- place live trees ---- */
    for (int i = 0; i < n_live; ++i) {
        int pos = idx[i];
        int r = pos / N, c = pos % N;
        grid[r][c] = TREE;
    }

    /* ---- place initially burning trees ---- */
    for (int i = 0; i < n_burning; ++i) {
        int pos = idx[n_live + i];
        int r = pos / N, c = pos % N;
        grid[r][c]     = BURNING;
        burn_age[r][c] = 0;
    }

    free(idx);
}

/* ============================== DISPLAY ============================== */

/**
 * @brief Render the current grid.
 *
 * Print mode uses `printf`; overlay mode uses `put()` from display.c.
 */
static void display_grid(void) {
    for (int r = 0; r < sim_size; ++r) {
        for (int c = 0; c < sim_size; ++c) {
            char ch;
            switch (grid[r][c]) {
                case EMPTY:  ch = ' '; break;
                case TREE:   ch = 'Y'; break;
                case BURNING:ch = '*'; break;
                case BURNED: ch = '.'; break;
                default:     ch = '?'; break;
            }
            if (print_mode) putchar(ch);
            else            put(ch);
        }
        if (print_mode) putchar('\n');
    }
}

/**
 * @brief Print the per-cycle statistics line(s).
 *
 * In print mode also prints "Fires are out." when appropriate.
 */
static void display_stats(int cycle, int changes, int total_changes) {
    printf("size %d, pCatch %.2f, density %.2f, pBurning %.2f, pNeighbor %.2f\n",
           sim_size,
           pCatch_pct / 100.0,
           density_pct / 100.0,
           pBurning_pct / 100.0,
           pNeighbor_pct / 100.0);
    printf("cycle %d, current changes %d, cumulative changes %d.\n",
           cycle, changes, total_changes);
}

/* ============================== SIMULATION UPDATE ============================== */

/**
 * @brief Advance the simulation one time step.
 *
 * Uses a copy of the previous state to avoid update-order skew.
 *
 * @param changes_out  Receives the number of cells that changed state.
 * @return 1 if any cell is still BURNING, 0 otherwise.
 */
static int update_simulation(int *changes_out) {
    const int N = sim_size;
    enum CellState old_grid[MAX_SIZE][MAX_SIZE];
    int            old_age [MAX_SIZE][MAX_SIZE];

    /* ---- copy current state ---- */
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            old_grid[r][c] = grid[r][c];
            old_age [r][c] = burn_age[r][c];
        }

    int changes      = 0;
    int any_burning  = 0;

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            enum CellState cur = old_grid[r][c];

            /* ---- empty or already burned – nothing to do ---- */
            if (cur == EMPTY || cur == BURNED) continue;

            /* ---- burning tree ---- */
            if (cur == BURNING) {
                int age = old_age[r][c];
                if (age >= BURN_DURATION - 1) {          /* final cycle */
                    grid[r][c]     = BURNED;
                    burn_age[r][c] = -1;
                    ++changes;
                } else {
                    grid[r][c]     = BURNING;
                    burn_age[r][c] = age + 1;
                    any_burning    = 1;
                }
                continue;
            }

            /* ---- live tree – may ignite ---- */
            if (cur == TREE) {
                int total_neigh = 0, burning_neigh = 0;

                for (int d = 0; d < 8; ++d) {
                    int nr = r + dr[d], nc = c + dc[d];
                    if (nr < 0 || nr >= N || nc < 0 || nc >= N) continue;
                    enum CellState ns = old_grid[nr][nc];
                    if (ns == TREE || ns == BURNING) {
                        ++total_neigh;
                        if (ns == BURNING) ++burning_neigh;
                    }
                }

                double proportion = total_neigh ?
                                    (double)burning_neigh / total_neigh : 0.0;

                int susceptible = (pNeighbor_pct == 0) ||
                                  (total_neigh && proportion * 100.0 >= pNeighbor_pct);

                if (susceptible && rand01() < pCatch_pct / 100.0) {
                    grid[r][c]     = BURNING;
                    burn_age[r][c] = 0;
                    ++changes;
                    any_burning    = 1;
                }
            }
        }
    }

    *changes_out = changes;
    return any_burning;
}