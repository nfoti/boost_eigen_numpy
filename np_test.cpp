#define WRAP_PYTHON 1
#if WRAP_PYTHON
#include <Python.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#endif

#include <iostream>
#include <complex>
#include <Eigen/Core>
#include <Eigen/QR>

// I somehow broke this.  Seems to be with putting s in function.

template<typename Derived, typename Scalar>
inline void scalemat(const Eigen::MatrixBase<Derived>& In,
             Eigen::MatrixBase<Derived>& Out, Scalar s)
{
    Out = s * In;
}

template<typename Derived>
inline double eigen_logdet(const Eigen::MatrixBase<Derived>& In)
{
    return In.householderQr().logAbsDeterminant();
}

#if WRAP_PYTHON
void dscal(PyObject *In, PyObject *Out, PyObject *Sc)
{
    npy_intp *n;
    double s;
    n = PyArray_DIMS(In);
    s = PyFloat_AsDouble(Sc);
    Eigen::Map<Eigen::VectorXd> eigIn((double*)PyArray_DATA(In),n[0]);
    Eigen::Map<Eigen::VectorXd> eigOut((double*)PyArray_DATA(Out),n[0]);
    return scalemat(eigIn, eigOut, s);
}

void zscal(PyObject *In, PyObject *Out, PyObject *Sc)
{
    npy_intp *n;
    n = PyArray_DIMS(In);
    std::complex<double> s(PyComplex_RealAsDouble(Sc), PyComplex_ImagAsDouble(Sc));
    Eigen::Map<Eigen::VectorXcd> eigIn((std::complex<double>*)PyArray_DATA(In),n[0]);
    Eigen::Map<Eigen::VectorXcd> eigOut((std::complex<double>*)PyArray_DATA(Out),n[0]);
    return scalemat(eigIn, eigOut, s);
}

boost::python::object dlogdet(PyObject *In)
{
    npy_intp *n;
    n = PyArray_DIMS(In);
    Eigen::Map<Eigen::MatrixXd> eigIn((double*)PyArray_DATA(In),n[0],n[1]);
    return boost::python::object(eigen_logdet(eigIn));
}

boost::python::object zlogdet(PyObject *In)
{
    npy_intp *n;
    n = PyArray_DIMS(In);
    Eigen::Map<Eigen::MatrixXcd> eigIn((std::complex<double>*)PyArray_DATA(In),n[0],n[1]);
    return boost::python::object(eigen_logdet(eigIn));
}



using namespace boost::python;
BOOST_PYTHON_MODULE(np_test)
{
    def("dscal", dscal);
    def("zscal", zscal);
    def("dlogdet", dlogdet);
    def("zlogdet", zlogdet);
}


#endif
