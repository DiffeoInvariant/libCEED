/// @file
/// Test scaling a vector
/// \test Test scaling of a vector
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedInt n;
  CeedScalar a[10];
  const CeedScalar *b;

  CeedInit(argv[1], &ceed);

  n = 10;
  CeedVectorCreate(ceed, n, &x);
  for (CeedInt i=0; i<n; i++)
    a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, a);

  CeedVectorScale(x, -0.5);

  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);
  for (CeedInt i=0; i<n; i++)
    if (fabs(b[i] + (10.0 + i)/2 ) > 1e-14)
      // LCOV_EXCL_START
      printf("Error in alpha x, computed: %f actual: %f\n", b[i],
             -(10.0 + i)/2);
  // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(x, &b);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
