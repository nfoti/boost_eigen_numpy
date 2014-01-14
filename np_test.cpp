#define WRAP_PYTHON 1
#if WRAP_PYTHON
#include <Python.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#endif

#include <iostream>
#include <iomanip>
#include <cstring>
#include <complex>
#include <random>
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
    auto n = PyArray_DIMS(In);
    auto s = PyFloat_AsDouble(Sc);
    auto eigIn = Eigen::Map<Eigen::VectorXd>{static_cast<double*>(PyArray_DATA(In)),n[0]};
    auto eigOut = Eigen::Map<Eigen::VectorXd>{static_cast<double*>(PyArray_DATA(Out)),n[0]};
    return scalemat(eigIn, eigOut, s);
}

void zscal(PyObject *In, PyObject *Out, PyObject *Sc)
{
    auto n = PyArray_DIMS(In);
    auto s = std::complex<double>{PyComplex_RealAsDouble(Sc), PyComplex_ImagAsDouble(Sc)};
    auto eigIn = Eigen::Map<Eigen::VectorXcd>{static_cast<std::complex<double>*>(PyArray_DATA(In)),n[0]};
    auto eigOut = Eigen::Map<Eigen::VectorXcd>{static_cast<std::complex<double>*>(PyArray_DATA(Out)),n[0]};
    return scalemat(eigIn, eigOut, s);
}

boost::python::object dlogdet(PyObject *In)
{
    // Can write this with a boost::python::object& and use In.ptr()
    auto n = PyArray_DIMS(In);
    auto eigIn = Eigen::Map<Eigen::MatrixXd>{static_cast<double*>(PyArray_DATA(In)),n[0],n[1]};
    return boost::python::object(eigen_logdet(eigIn));
}

boost::python::object zlogdet(PyObject *In)
{
    // Can write this with a boost::python::object& and use In.ptr()
    auto n = PyArray_DIMS(In);
    auto eigIn = Eigen::Map<Eigen::MatrixXcd>{static_cast<std::complex<double>*>(PyArray_DATA(In)),n[0],n[1]};
    return boost::python::object(eigen_logdet(eigIn));
}

boost::python::object randvec(PyObject *Sz)
{
    // Can write this with a boost::python::object& and use Sz.ptr()
    auto seed = 8675309;
    auto gen = std::mt19937_64{seed};
    auto n = PyInt_AsLong(Sz);
    auto ndist = std::normal_distribution<double>{0., 1.};
    //auto normal = [&](double){return ndist(gen);};
    auto normal = [&](){return ndist(gen);};

    // Allocate our data and generate from standard normal
#define NF_TESTING 1
    auto data = static_cast<double*>(new double[n]);
    for (int i = 0; i < n; i++)
    {
        data[i] = normal();
#if NF_TESTING
        std::cout << data[i] << " ";
#endif
    }
#if NF_TESTING
    std::cout << "\n";
#endif

    npy_intp *shape = &n;
    npy_intp strides[1] = {sizeof(double)};

    auto array = boost::python::handle<>{PyArray_New(&PyArray_Type, 1, shape,
                                         NPY_DOUBLE, strides, data,
                                         sizeof(double), NPY_ARRAY_CARRAY, NULL)};
    return boost::python::object(array);
}

boost::python::list list_of_ndarray()
{
    boost::python::list l;
    for (int i = 0; i < 5; i++)
    {
        npy_intp shape[1] = {i};
        npy_intp strides[1] = {sizeof(double)};
        auto data = static_cast<double*>(new double[i]);
        for (int j = 0; j < i; j++)
            data[j] = i;
        auto array = boost::python::handle<>{PyArray_New(&PyArray_Type, 1, shape,
                                             NPY_DOUBLE, strides, data,
                                             sizeof(double), NPY_ARRAY_CARRAY, NULL)};
        l.append(boost::python::object(array));
    }
    return l;
}

void input_list_of_ndarray(boost::python::list& List)
{
    for (int i = 0; i < boost::python::len(List); i++)
    {
        boost::python::object pobj = List[i];
        auto l = pobj.ptr();
        auto nd = PyArray_NDIM(l);
        auto n = PyArray_DIMS(l);
        if (nd == 2)
        {
            auto X = Eigen::Map<Eigen::MatrixXd>{static_cast<double*>(PyArray_DATA(l)),n[0],n[1]};
            std::cout << "Matrix: " << X << "\n";
        }
        else
            std::cout << "ndarray must be two dimensional\n";
    }
}

using namespace boost::python;
BOOST_PYTHON_MODULE(np_test)
{
    // To use C-API memory funtions.  This was actually very important
    import_array();

    def("dscal", dscal);
    def("zscal", zscal);
    def("dlogdet", dlogdet);
    def("zlogdet", zlogdet);
    def("randvec", randvec);
    def("list_of_ndarray", list_of_ndarray);
    def("input_list_of_ndarray", input_list_of_ndarray);
}


#endif
