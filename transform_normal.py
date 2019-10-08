import numpy as np
import matplotlib.pyplot as plt

# This program computes the adjoint and inverse of a matrix M, then compares the difference when you use both to transform the normal of the line between point1 and point2

# Positive determinant matrix
# M = np.matrix([[6, 0, 0], [0, 2, 0], [0, 0, 1]])

# Negative determinant matrix
M = np.matrix([[-6, 0, 0], [0, 2, 0], [0, 0, 1]])

point1 = np.array([1, 0, 1])
point2 = np.array([0, 1, 1])

# ====
print("Original matrix")
print(M)

det = np.linalg.det(M)
print("Determinant")
print(det)

adjoint = np.zeros_like(M)
x, y = adjoint.shape
for i in range(0, x):
    for j in range(0, y):
        minor = np.delete(np.delete(M,i,axis=0), j, axis=1)
        print(minor)
        sign = (i + j) % 2 * -2 + 1
        print(sign)
        # note that the indices i and j are reversed because the adjoint
        # is the transpose of the cofactor
        adjoint[j, i] = sign * np.linalg.det(minor)

print("Adjoint")
print(adjoint)

inverse = np.linalg.inv(M)
print("Inverse")
print(inverse)

print("Calculated inverse from adjoint")
print(np.divide(adjoint, det))

# transform these points using M
new_point1 = M.dot(point1)
new_point2 = M.dot(point2)

# plot the old line, new line, old normal, and new normal
old_x_list = [point1[0], point2[0]]
old_y_list = [point1[1], point2[1]]
new_x_list = [new_point1[0, 0], new_point2[0, 0]]
new_y_list = [new_point1[0, 1], new_point2[0, 1]]
slope = (old_x_list[1] - old_x_list[0])/(old_y_list[1] - old_y_list[0])
normal = np.array([1, -1/slope, 0])

def normalize(vector):
    return vector / np.linalg.norm(vector)
normal = normalize(normal)

def get_midpoint(x_list, y_list):
    return np.array([(x_list[1] + x_list[0])/2, (y_list[1] + y_list[0])/2, 1])

normal_origin = get_midpoint(old_x_list, old_y_list)

normal_end = normal_origin + normal
normal_x_list = [normal_origin[0], normal_end[0]]
normal_y_list = [normal_origin[1], normal_end[1]]
print(normal_origin, normal, normal_end)

def get_transformed_normal_render_points(normal, t_matrix, x_list, y_list):
    new_normal = normalize(t_matrix.dot(normal))
    new_normal_origin = get_midpoint(x_list, y_list)
    new_normal_end = new_normal_origin + new_normal
    adjoint_normal_x_list = [new_normal_origin[0], new_normal_end[0, 0]]
    adjoint_normal_y_list = [new_normal_origin[1], new_normal_end[0, 1]]
    return adjoint_normal_x_list, adjoint_normal_y_list

adjoint_normal_x_list, adjoint_normal_y_list = get_transformed_normal_render_points(normal, adjoint, new_x_list, new_y_list)

inverse_normal_x_list, inverse_normal_y_list = get_transformed_normal_render_points(normal, inverse, new_x_list, new_y_list)


plt.plot(old_x_list, old_y_list, "ro-")
plt.plot(normal_x_list, normal_y_list, "bo-")
plt.plot(new_x_list, new_y_list, "ro-")
plt.plot(adjoint_normal_x_list, adjoint_normal_y_list, "bo-")
plt.plot(inverse_normal_x_list, inverse_normal_y_list, "go-")
plt.axis('scaled')
plt.show()

