//#define Py_LIMITED_API 3 // No ABI3 conformant. Keep in mind C-API NumPy is no ABI3
#include "tdigest.h"

/*
https://docs.python.org/3/c-api/
https://docs.python.org/3/c-api/stable.html
ABI3              PEP 384
PyModuleDef,      PEP 3121
PyCapsule,        PEP 3118/3121
PY_SSIZE_T_CLEAN, PEP 353
Py_ssize_t,       PEP 353

https://numpy.org/doc/1.25/reference/c-api/index.html
PyArrayObject,    C-API NumPy
PyArray_FROM_OTF, C-API NumPy
*/



#include <numpy/arrayobject.h>
PyObject* py_add_batch_zerocopy(PyObject *self, PyObject *args) {
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

