// Author: Nic Olsen

#include <hip/hip_runtime.h>
#include <iostream>

#include "reduce.h"
//#include "reduce_impl.h"
// either GCPUSRB or GSRB should be defined, but not both
#ifdef GCPUSRB
  #include "reduce_gcpusrb.h"
#else /* GSRB */
  #include "reduce_gsrb.h"
#endif

int main(int argc, char* argv[]) {
    if (argc != 3) {
      fprintf(stderr, "ERROR: ./reduce <N> <thrPerWG>\n");
      fprintf(stderr, "where:\n");
      fprintf(stderr, "\t<N>: number of elements\n");
      fprintf(stderr, "\t<thrPerWG>: # of threads per WG\n");
      return -1;
    }
    int N = std::atoi(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]);

    int* arr = new int[N];

    for (int i = 0; i < N; i++) {
        arr[i] = 1;
    }
    int correct_sum = N;


    int sum = reduce(arr, N, threads_per_block);

    // Get the elapsed time in milliseconds
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Correct sum: " << correct_sum << std::endl;

    delete[] arr;
    return 0;
}
