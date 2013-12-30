
import numpy as np
import np_test

# dscal and zscal (scale vector by scalar)
print "dscal"
print "-----------"
x = np.ones(5)
y = np.empty(5)
s = 4.
np_test.dscal(x, y, s)
print "numpy: ",
print x*s
print "np_test: ",
print y
print "Both the same: ",
print np.allclose(x*s, y)

print "zscal"
print "-----------"
x = np.arange(0,10,2) + np.arange(1,11,2)*1.j
y = np.empty(5, dtype='complex128')
s = 2. + 3.j
np_test.zscal(x, y, s)
print "numpy: ",
print x*s
print "np_test: ",
print y
print "Both the same? ",
print np.allclose(x*s, y)

# logdet for doubles and complex doubles
print "logdet (double)"
print "-----------"
A = np.random.randn(5,5)
print "numpy: ",
print np.linalg.slogdet(A)[1]
print "np_test: ",
print np_test.dlogdet(A)
print "Both the same? ",
print np.allclose(np_test.dlogdet(A), np.linalg.slogdet(A)[1])

print "logdet (complex double)"
print "-----------"
A = np.random.randn(5,5) + np.random.randn(5,5)*1.j
print "numpy: ",
print np.linalg.slogdet(A)[1]
print "np_test: ",
print np_test.zlogdet(A)
print "Both the same? ",
print np.allclose(np_test.zlogdet(A), np.linalg.slogdet(A)[1])

