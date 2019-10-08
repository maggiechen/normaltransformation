import numpy as np

# m = np.matrix([[-3, 2, -5], [-1, 0, -2], [3, -4, 1]])
m = np.matrix([[-3, 0, 0], [0, 2, 0], [0, 0, 1]])
print(m)

det = np.linalg.det(m)
print("Determinant")
print(det)

adjoint = np.zeros_like(m)
x, y = adjoint.shape
for i in range(0, x):
    for j in range(0, y):
        minor = np.delete(np.delete(m,i,axis=0), j, axis=1)
        print(minor)
        sign = (i + j) % 2 * -2 + 1
        print(sign)
        # note that the indices i and j are reversed because the adjoint
        # is the transpose of the cofactor
        adjoint[j, i] = sign * np.linalg.det(minor)

print("Adjoint")
print(adjoint)

inverse = np.linalg.inv(m)
print("Inverse")
print(inverse)

print("Calculated inverse from adjoint")
print(np.divide(adjoint, det))

