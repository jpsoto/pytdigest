#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "tdigest.c"

#define CAPSULE_NAME "pytdigest.tdigest"

static void capsule_destructor(PyObject *capsule) {
    tdigest_t *td = PyCapsule_GetPointer(capsule, CAPSULE_NAME);
    if (td) td_free(td);
}

static tdigest_t* get_td(PyObject *capsule) {
    return (tdigest_t*)PyCapsule_GetPointer(capsule, CAPSULE_NAME);
}

/* create(delta) */
static PyObject* py_create(PyObject *self, PyObject *args) {
    double delta;
    if (!PyArg_ParseTuple(args, "d", &delta)) return NULL;
    tdigest_t *td = td_new(delta);
    if (!td) return PyErr_NoMemory();
    return PyCapsule_New(td, CAPSULE_NAME, capsule_destructor);
}

/* reset(td) */
static PyObject* py_reset(PyObject *self, PyObject *args) {
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    td_reset(get_td(cap));
    Py_RETURN_NONE;
}

/* add(td, x, w) */
static PyObject* py_add(PyObject *self, PyObject *args) {
    PyObject *cap; double x,w;
    if (!PyArg_ParseTuple(args,"Odd",&cap,&x,&w)) return NULL;
    td_add(get_td(cap),x,w);
    Py_RETURN_NONE;
}

/* merge(into, from) */
static PyObject* py_merge(PyObject *self, PyObject *args) {
    PyObject *a,*b;
    if (!PyArg_ParseTuple(args,"OO",&a,&b)) return NULL;
    td_merge(get_td(a),get_td(b));
    Py_RETURN_NONE;
}

/* value_at(td,q) */
static PyObject* py_value_at(PyObject *self, PyObject *args) {
    PyObject *cap; double q;
    if (!PyArg_ParseTuple(args,"Od",&cap,&q)) return NULL;
    return PyFloat_FromDouble(td_value_at(get_td(cap),q));
}

/* quantile_of(td,x) */
static PyObject* py_quantile_of(PyObject *self, PyObject *args) {
    PyObject *cap; double x;
    if (!PyArg_ParseTuple(args,"Od",&cap,&x)) return NULL;
    return PyFloat_FromDouble(td_quantile_of(get_td(cap),x));
}

/* trimmed_mean(td,lo,hi) */
static PyObject* py_trimmed_mean(PyObject *self, PyObject *args) {
    PyObject *cap; double lo,hi;
    if (!PyArg_ParseTuple(args,"Odd",&cap,&lo,&hi)) return NULL;
    return PyFloat_FromDouble(td_trimmed_mean(get_td(cap),lo,hi));
}

/* totals */
static PyObject* py_total_weight(PyObject *self, PyObject *args) {
    PyObject *cap;
    if (!PyArg_ParseTuple(args,"O",&cap)) return NULL;
    return PyFloat_FromDouble(td_total_weight(get_td(cap)));
}

static PyObject* py_total_sum(PyObject *self, PyObject *args) {
    PyObject *cap;
    if (!PyArg_ParseTuple(args,"O",&cap)) return NULL;
    return PyFloat_FromDouble(td_total_sum(get_td(cap)));
}

/* scale_weight */
static PyObject* py_scale_weight(PyObject *self, PyObject *args) {
    PyObject *cap; double f;
    if (!PyArg_ParseTuple(args,"Od",&cap,&f)) return NULL;
    td_scale_weight(get_td(cap),f);
    Py_RETURN_NONE;
}

/* shift */
static PyObject* py_shift(PyObject *self, PyObject *args) {
    PyObject *cap; double s;
    if (!PyArg_ParseTuple(args,"Od",&cap,&s)) return NULL;
    td_shift(get_td(cap),s);
    Py_RETURN_NONE;
}

/* add_batch(td, means, weights) */
static PyObject* py_add_batch(PyObject *self, PyObject *args) {
    PyObject *cap,*ml,*wl;
    if (!PyArg_ParseTuple(args,"OOO",&cap,&ml,&wl)) return NULL;

    Py_ssize_t n = PySequence_Length(ml);
    if (n != PySequence_Length(wl))
        return PyErr_Format(PyExc_ValueError,"size mismatch");

    double *means = malloc(sizeof(double)*n);
    double *weights = malloc(sizeof(double)*n);

    for (Py_ssize_t i=0;i<n;i++) {
        means[i] = PyFloat_AsDouble(PySequence_GetItem(ml,i));
        weights[i] = PyFloat_AsDouble(PySequence_GetItem(wl,i));
    }

    td_add_batch(get_td(cap),(int)n,means,weights);
    free(means); free(weights);
    Py_RETURN_NONE;
}

#include <numpy/arrayobject.h>
static PyObject* py_add_batch_TBC(PyObject *self, PyObject *args) {
    PyObject *cap, *x_obj, *w_obj;
    if (!PyArg_ParseTuple(args,"OOO",&cap,&x_obj,&w_obj)) return NULL;

    PyArrayObject *x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *w_array = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (x_array == NULL || w_array == NULL) return NULL;

    npy_intp n = PyArray_SIZE(x_array);
    if (PyArray_SIZE(w_array) != n) {
        Py_XDECREF(x_array);
        Py_XDECREF(w_array);
        return PyErr_Format(PyExc_ValueError,"size mismatch");
    }

    double *means = (double*)PyArray_DATA(x_array);
    double *weights = (double*)PyArray_DATA(w_array);

    td_add_batch(get_td(cap), (int)n, means, weights);

    Py_XDECREF(x_array);
    Py_XDECREF(w_array);
    Py_RETURN_NONE;
}


/* batch CDF */
static PyObject* py_cdf_batch(PyObject *self, PyObject *args) {
    PyObject *cap,*vals;
    if (!PyArg_ParseTuple(args,"OO",&cap,&vals)) return NULL;

    Py_ssize_t n = PySequence_Length(vals);
    double *in = malloc(sizeof(double)*n);
    double *out = malloc(sizeof(double)*n);

    for (Py_ssize_t i=0;i<n;i++)
        in[i]=PyFloat_AsDouble(PySequence_GetItem(vals,i));

    td_cdf_batch(get_td(cap),(int)n,in,out);

    PyObject *list=PyList_New(n);
    for (Py_ssize_t i=0;i<n;i++)
        PyList_SET_ITEM(list,i,PyFloat_FromDouble(out[i]));

    free(in); free(out);
    return list;
}

/* inverse CDF batch */
static PyObject* py_inverse_cdf_batch(PyObject *self, PyObject *args) {
    PyObject *cap,*qs;
    if (!PyArg_ParseTuple(args,"OO",&cap,&qs)) return NULL;

    Py_ssize_t n = PySequence_Length(qs);
    double *in = malloc(sizeof(double)*n);
    double *out = malloc(sizeof(double)*n);

    for (Py_ssize_t i=0;i<n;i++)
        in[i]=PyFloat_AsDouble(PySequence_GetItem(qs,i));

    td_inverse_cdf_batch(get_td(cap),(int)n,in,out);

    PyObject *list=PyList_New(n);
    for (Py_ssize_t i=0;i<n;i++)
        PyList_SET_ITEM(list,i,PyFloat_FromDouble(out[i]));

    free(in); free(out);
    return list;
}

/* get_centroids */
static PyObject* py_get_centroids(PyObject *self, PyObject *args) {
    PyObject *cap;
    if (!PyArg_ParseTuple(args,"O",&cap)) return NULL;

    tdigest_t *td = get_td(cap);
    int n = td->num_merged + td->num_unmerged;

    double *buf = malloc(sizeof(double)*2*n);
    td_get_centroids(td,buf);

    PyObject *list=PyList_New(n);
    for (int i=0;i<n;i++) {
        PyObject *pair=PyTuple_Pack(
            2,
            PyFloat_FromDouble(buf[2*i]),
            PyFloat_FromDouble(buf[2*i+1])
        );
        PyList_SET_ITEM(list,i,pair);
    }
    free(buf);
    return list;
}

/* of_centroids */
static PyObject* py_of_centroids(PyObject *self, PyObject *args) {
    double delta; PyObject *list;
    if (!PyArg_ParseTuple(args,"dO",&delta,&list)) return NULL;

    Py_ssize_t n = PySequence_Length(list);
    double *buf = malloc(sizeof(double)*2*n);

    for (Py_ssize_t i=0;i<n;i++) {
        PyObject *pair = PySequence_GetItem(list,i);
        buf[2*i] = PyFloat_AsDouble(PySequence_GetItem(pair,0));
        buf[2*i+1] = PyFloat_AsDouble(PySequence_GetItem(pair,1));
    }

    tdigest_t *td = td_of_centroids(delta,(int)n,buf);
    free(buf);

    return PyCapsule_New(td,CAPSULE_NAME,capsule_destructor);
}

static PyMethodDef Methods[] = {
    {"create",py_create,METH_VARARGS,""},
    {"reset",py_reset,METH_VARARGS,""},
    {"add",py_add,METH_VARARGS,""},
    {"merge",py_merge,METH_VARARGS,""},
    {"value_at",py_value_at,METH_VARARGS,""},
    {"quantile_of",py_quantile_of,METH_VARARGS,""},
    {"trimmed_mean",py_trimmed_mean,METH_VARARGS,""},
    {"total_weight",py_total_weight,METH_VARARGS,""},
    {"total_sum",py_total_sum,METH_VARARGS,""},
    {"scale_weight",py_scale_weight,METH_VARARGS,""},
    {"shift",py_shift,METH_VARARGS,""},
    {"add_batch",py_add_batch,METH_VARARGS,""},
    {"cdf_batch",py_cdf_batch,METH_VARARGS,""},
    {"inverse_cdf_batch",py_inverse_cdf_batch,METH_VARARGS,""},
    {"get_centroids",py_get_centroids,METH_VARARGS,""},
    {"of_centroids",py_of_centroids,METH_VARARGS,""},
    {NULL,NULL,0,NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_tdigest",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC PyInit__tdigest(void) {
    return PyModule_Create(&module);
}

