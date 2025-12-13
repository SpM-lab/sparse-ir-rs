#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sparseir/sparseir.h>
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

// Get maximum memory usage in MB (cumulative maximum)
static long get_max_memory_usage_mb(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
        return usage.ru_maxrss / (1024 * 1024); // On macOS, ru_maxrss is in bytes
#else
        return usage.ru_maxrss / 1024; // On Linux, ru_maxrss is in KB
#endif
    }
    return -1;
}

// Get current memory usage in MB (actual current usage)
static long get_current_memory_usage_mb(void) {
#ifdef __APPLE__
    struct task_basic_info info;
    mach_msg_type_number_t size = sizeof(info);
    kern_return_t kerr = task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &size);
    if (kerr == KERN_SUCCESS) {
        return info.resident_size / (1024 * 1024); // Convert bytes to MB
    }
    return -1;
#else
    // On Linux, use /proc/self/status
    FILE *file = fopen("/proc/self/status", "r");
    if (file) {
        char line[128];
        while (fgets(line, sizeof(line), file)) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                long mem_kb;
                sscanf(line, "VmRSS: %ld", &mem_kb);
                fclose(file);
                return mem_kb / 1024; // Convert KB to MB
            }
        }
        fclose(file);
    }
    return -1;
#endif
}

// Print memory usage (both current and maximum)
static void print_memory_usage(const char *label) {
    long max_mem_mb = get_max_memory_usage_mb();
    long current_mem_mb = get_current_memory_usage_mb();
    
    if (current_mem_mb >= 0) {
        printf("[MEMORY] %s: current=%ld MB (%.2f GB), max=%ld MB (%.2f GB)\n", 
               label, current_mem_mb, current_mem_mb / 1024.0, 
               max_mem_mb, max_mem_mb / 1024.0);
    } else {
        printf("[MEMORY] %s: max=%ld MB (%.2f GB) [current unavailable]\n", 
               label, max_mem_mb, max_mem_mb / 1024.0);
    }
}

int main(int argc, char *argv[])
{
    double beta = 1e+5;
    double omega_max = 1.0;
    double epsilon = 1e-10;

    // Parse command line arguments
    if (argc > 1) {
        beta = atof(argv[1]);
    }
    if (argc > 2) {
        epsilon = atof(argv[2]);
    }
    if (argc > 3) {
        omega_max = atof(argv[3]);
    }

    printf("========================================\n");
    printf("SVE Memory Test\n");
    printf("========================================\n");
    printf("beta: %e\n", beta);
    printf("omega_max: %f\n", omega_max);
    printf("epsilon: %e\n", epsilon);
    printf("lambda = beta * omega_max = %e\n", beta * omega_max);
    printf("========================================\n\n");

    int32_t status;
    long initial_mem_mb, after_kernel_mem_mb, after_sve_mem_mb;

    print_memory_usage("Initial");
    initial_mem_mb = get_current_memory_usage_mb();
    if (initial_mem_mb < 0) {
        initial_mem_mb = get_max_memory_usage_mb();
    }

    // Step 1: Create kernel
    printf("\n[STEP 1] Creating kernel (lambda = %e)...\n", beta * omega_max);
    spir_kernel *kernel = spir_logistic_kernel_new(beta * omega_max, &status);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        printf("[ERROR] Kernel creation failed with status: %d\n", status);
        return 1;
    }
    if (kernel == NULL) {
        printf("[ERROR] Kernel is NULL\n");
        return 1;
    }
    printf("[OK] Kernel created\n");
    print_memory_usage("After kernel creation");
    after_kernel_mem_mb = get_current_memory_usage_mb();
    if (after_kernel_mem_mb < 0) {
        after_kernel_mem_mb = get_max_memory_usage_mb();
    }

    // Step 2: Compute SVE
    printf("\n[STEP 2] Computing SVE (epsilon = %e)...\n", epsilon);
    int lmax = -1;
    int n_gauss = -1;
    int Twork = SPIR_TWORK_AUTO;  // Auto-select: FLOAT64 for epsilon >= 1e-8, FLOAT64X2 for epsilon < 1e-8
    printf("Twork: %d (SPIR_TWORK_AUTO)\n", Twork);
    
    spir_sve_result *sve_logistic = spir_sve_result_new(
        kernel, epsilon, lmax, n_gauss, Twork, &status);
    
    if (status != SPIR_COMPUTATION_SUCCESS) {
        printf("[ERROR] SVE computation failed with status: %d\n", status);
        spir_kernel_release(kernel);
        return 1;
    }
    if (sve_logistic == NULL) {
        printf("[ERROR] SVE result is NULL\n");
        spir_kernel_release(kernel);
        return 1;
    }
    printf("[OK] SVE computed\n");
    print_memory_usage("After SVE computation");
    after_sve_mem_mb = get_current_memory_usage_mb();
    if (after_sve_mem_mb < 0) {
        after_sve_mem_mb = get_max_memory_usage_mb();
    }

    // Get SVE result information if available
    printf("\n[INFO] SVE computation completed successfully\n");
    printf("Memory increase from initial: %.2f GB (%.2f MB)\n", 
           (after_sve_mem_mb - initial_mem_mb) / 1024.0,
           (double)(after_sve_mem_mb - initial_mem_mb));
    printf("Memory increase from kernel creation: %.2f GB (%.2f MB)\n", 
           (after_sve_mem_mb - after_kernel_mem_mb) / 1024.0,
           (double)(after_sve_mem_mb - after_kernel_mem_mb));

    // Clean up
    printf("\n[CLEANUP] Releasing resources...\n");
    spir_sve_result_release(sve_logistic);
    print_memory_usage("After releasing SVE result");
    
    spir_kernel_release(kernel);
    print_memory_usage("After releasing kernel");

    printf("\n========================================\n");
    printf("SVE Memory Test Completed\n");
    printf("========================================\n");

    return 0;
}
