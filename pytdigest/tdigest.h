#ifndef TDIGEST_H
#define TDIGEST_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

#define PY_SSIZE_T_CLEAN // https://docs.python.org/3/c-api/arg.html#strings-and-buffers
#include <Python.h>

// Non ABI functions, see tdigest_noabi.c
PyObject* py_add_batch_zerocopy(PyObject *self, PyObject *args);

typedef struct tdigest tdigest_t;

typedef struct centroid {
    double mean;
    double weight;
} centroid_t;


/* Constructors / lifecycle */
DLL_EXPORT tdigest_t *td_new(double compression);
DLL_EXPORT void td_free(tdigest_t *h);
DLL_EXPORT void td_reset(tdigest_t *h);

#define PY_CAPSULE_NAME "pytdigest.tdigest"
static void capsule_destructor(PyObject *capsule) {
    tdigest_t *td = PyCapsule_GetPointer(capsule, PY_CAPSULE_NAME);
    if (td) td_free(td);
}
static tdigest_t* get_td(PyObject *capsule) {
    return (tdigest_t*)PyCapsule_GetPointer(capsule, PY_CAPSULE_NAME);
}

/* Core operations */
DLL_EXPORT void td_add(tdigest_t *h, double val, double count);
DLL_EXPORT void td_add_batch(tdigest_t *h, int num_values,
                             double *means, double *weights);

DLL_EXPORT void td_merge(tdigest_t *into, tdigest_t *from);
DLL_EXPORT void merge(tdigest_t *h);   /* explicit compression */


/* Queries */
DLL_EXPORT double td_value_at(tdigest_t *h, double q);
DLL_EXPORT double td_quantile_of(tdigest_t *h, double val);
DLL_EXPORT double td_trimmed_mean(tdigest_t *h, double lo, double hi);

DLL_EXPORT double td_total_weight(tdigest_t *h);
DLL_EXPORT double td_total_sum(tdigest_t *h);


/* Transformations */
DLL_EXPORT void td_scale_weight(tdigest_t *h, double factor);
DLL_EXPORT void td_shift(tdigest_t *h, double shift);


/* Batch queries */
DLL_EXPORT void td_cdf_batch(tdigest_t *h, int count,
                             const double *values,
                             double *quantiles);

DLL_EXPORT void td_inverse_cdf_batch(tdigest_t *h, int count,
                                     const double *quantiles,
                                     double *values);


/* Centroid access */
DLL_EXPORT centroid_t *td_get_centroid(tdigest_t *h, int i);
DLL_EXPORT int td_num_centroids(tdigest_t *h);

DLL_EXPORT void td_get_centroids(tdigest_t *h, double *centroids);
DLL_EXPORT void td_fill_centroids(tdigest_t *h, int num_centroids,
                                  double *centroids);

DLL_EXPORT tdigest_t *td_of_centroids(double delta,
                                      int num_centroids,
                                      double *centroids);

#ifdef __cplusplus
}
#endif

#endif /* TDIGEST_H */

