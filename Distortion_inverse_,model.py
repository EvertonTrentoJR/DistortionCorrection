##############################################
# THESE CODE IS USED TO CALCULATE
# THE INVERSE DISTORTION MODEL
# USING THIN-PLATE SPLINE ALGORITHM
#
# E. tRENTO jR.


# import
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import cg
import time


def quadratic_regression_3d(pixel_map_mX, pixel_map_mY, pixel_map_mZ, spatial_map_mX, spatial_map_mY, spatial_map_mZ):
    pixel_map_mX_out = pixel_map_mX
    pixel_map_mY_out = pixel_map_mY
    pixel_map_mZ_out = pixel_map_mZ
    spatial_map_mX_out = spatial_map_mX
    spatial_map_mY_out = spatial_map_mY
    spatial_map_mZ_out = spatial_map_mZ

    rows = spatial_map_mX_out.size
    cols = 10
    ls_X = np.zeros((rows, cols))
    for i in range(rows):
        ls_X[i, :] = np.array(
            [1, spatial_map_mX_out[i], spatial_map_mY_out[i], spatial_map_mZ_out[i],
             spatial_map_mX_out[i] * spatial_map_mY_out[i],
             spatial_map_mX_out[i] * spatial_map_mZ_out[i], spatial_map_mY_out[i] * pixel_map_mZ_out[i],
             spatial_map_mX_out[i] ** 2, spatial_map_mY_out[i] ** 2, spatial_map_mZ_out[i] ** 2])
    ls_X_X = cg(np.matmul(ls_X.transpose(), ls_X),
                np.matmul(ls_X.transpose(), pixel_map_mX_out))
    a_coef = ls_X_X[0]
    ls_X_X = cg(np.matmul(ls_X.transpose(), ls_X),
                np.matmul(ls_X.transpose(), pixel_map_mY_out))
    b_coef = ls_X_X[0]
    ls_X_X = cg(np.matmul(ls_X.transpose(), ls_X),
                np.matmul(ls_X.transpose(), pixel_map_mZ_out))
    c_coef = ls_X_X[0]

    print('--------3D Quadratic Regression Coef--------')
    print('Coefficients for x: ' + str(a_coef))
    print('Coefficients for y: ' + str(b_coef))
    print('Coefficients for z: ' + str(c_coef))
    print('--------------------------------------------')

    Quadratic_coef = np.array([a_coef, b_coef, c_coef])

    x_model = np.matmul(ls_X, a_coef)
    y_model = np.matmul(ls_X, b_coef)
    z_model = np.matmul(ls_X, c_coef)

    # residual error
    residual = np.sum(
        (x_model - pixel_map_mX_out) ** 2 + (y_model - pixel_map_mY_out) ** 2 + (z_model - pixel_map_mZ_out) ** 2)
    print('QR Sum of squared errors (SSE): ' + str(residual))

    return Quadratic_coef, x_model, y_model, z_model


def thinplatespline(x_distort, y_distort, z_distort, x_true, y_true, z_true):
    # these function GENERATES DISTORTION COEFFICIENTS
    # distortion model - correct points fit to distort ones ->>>>> (x_distort, y_distort, z_distort, x_true, y_true, z_true)
    # inverse distortion model - distorted points fit to correct ones  ->>>>> (x_true, y_true, z_true, x_distort, y_distort, z_distort)

    N = len(x_true)
    D = 3  # number of spatial dimensionsl

    # Radial Basis Function U:
    def U(r):
        RBF = np.power(r, 2) * np.log(r)
        RBF[r <= 0] = 0.
        return RBF

    def dist(pt0, pt1):
        return cdist(pt0, pt1)

    def f(pt_dist):
        result_x = wa_x[N] + wa_x[N + 1] * pt_dist[:, 0] + wa_x[N + 2] * pt_dist[:, 1] + wa_x[N + 3] * pt_dist[:, 2] + \
                   np.dot(np.transpose(wa_x[:N]), U(dist(pt_true, pt_dist)))
        result_y = wa_y[N] + wa_y[N + 1] * pt_dist[:, 0] + wa_y[N + 2] * pt_dist[:, 1] + wa_y[N + 3] * pt_dist[:, 2] + \
                   np.dot(np.transpose(wa_y[:N]), U(dist(pt_true, pt_dist)))
        result_z = wa_z[N] + wa_z[N + 1] * pt_dist[:, 0] + wa_z[N + 2] * pt_dist[:, 1] + wa_z[N + 3] * pt_dist[:, 2] + \
                   np.dot(np.transpose(wa_z[:N]), U(dist(pt_true, pt_dist)))
        return result_x, result_y, result_z,

    P = np.zeros((N, D + 1))
    P[:, 0] = 1
    P[:, 1] = x_true
    P[:, 2] = y_true
    P[:, 3] = z_true

    vo_x = np.zeros((N + D + 1, 1), dtype=float)
    vo_y = np.zeros((N + D + 1, 1), dtype=float)
    vo_z = np.zeros((N + D + 1, 1), dtype=float)
    vo_x[:N, 0] = x_distort
    vo_y[:N, 0] = y_distort
    vo_z[:N, 0] = z_distort

    K = np.zeros((N, N), dtype=float)
    pt0 = np.zeros((N, 3), dtype=float)
    pt1 = np.zeros((N, 3), dtype=float)

    pt0[:, 0] = x_true
    pt0[:, 1] = y_true
    pt0[:, 2] = z_true
    pt1[:, 0] = x_true
    pt1[:, 1] = y_true
    pt1[:, 2] = z_true

    K = U(dist(pt0, pt1))

    L = np.zeros((N + D + 1, N + D + 1))
    L[:N, :N] = K
    L[:N, N:] = P
    L[N:, :N] = P.transpose()

    L_inv = pinv(L.transpose() @ L) @ L.transpose()
    wa_x = L_inv @ vo_x
    wa_y = L_inv @ vo_y
    wa_z = L_inv @ vo_z

    # plot
    x_TPS = np.zeros(N)
    y_TPS = np.zeros(N)
    z_TPS = np.zeros(N)

    pt_true = np.zeros((N, 3), dtype=float)
    pt_dist = np.zeros((N, 3), dtype=float)

    pt_true[:, 0] = x_true
    pt_true[:, 1] = y_true
    pt_true[:, 2] = z_true

    pt_dist[:, 0] = x_true
    pt_dist[:, 1] = y_true
    pt_dist[:, 2] = z_true

    x_TPS, y_TPS, z_TPS = f(pt_dist)

    sse = np.sum((x_distort - x_TPS) ** 2 + (y_distort - y_TPS) ** 2 + (z_distort - z_TPS) ** 2)
    print('Sum of squared errors (SSE): ' + str(sse))

    TPS_coef = np.array([wa_x, wa_y, wa_z])

    return TPS_coef, x_TPS, y_TPS, z_TPS


path_train_mean_Dense = 'C:/Users/treen/PycharmProjects/pythonProject/dense_point_cloud/setup_paper/tanque/medias/'
for i in range(1, 51):
    exec(f'pixel_map_X_mean{i}= np.load(path_train_mean_Dense + f"Dense_pixel_map_X_mean{i}.npy")')
    exec(f'pixel_map_Y_mean{i}= np.load(path_train_mean_Dense + f"Dense_pixel_map_Y_mean{i}.npy")')
    exec(f'pixel_map_Z_mean{i}= np.load(path_train_mean_Dense + f"Dense_pixel_map_Z_mean{i}.npy")')

def train_data():
    # plot all train mean data
    camera_mean_x = np.concatenate((pixel_map_X_mean1.ravel(), pixel_map_X_mean2.ravel(), pixel_map_X_mean3.ravel(),
                                    pixel_map_X_mean4.ravel(), pixel_map_X_mean5.ravel(), pixel_map_X_mean6.ravel(),
                                    pixel_map_X_mean7.ravel(), pixel_map_X_mean8.ravel(), pixel_map_X_mean9.ravel(),
                                    pixel_map_X_mean10.ravel(),
                                    pixel_map_X_mean11.ravel(), pixel_map_X_mean12.ravel(), pixel_map_X_mean13.ravel(),
                                    pixel_map_X_mean14.ravel(), pixel_map_X_mean15.ravel(), pixel_map_X_mean16.ravel(),
                                    pixel_map_X_mean17.ravel(), pixel_map_X_mean18.ravel(), pixel_map_X_mean19.ravel(),
                                    pixel_map_X_mean20.ravel(),
                                    pixel_map_X_mean21.ravel(), pixel_map_X_mean22.ravel(), pixel_map_X_mean23.ravel(),
                                    pixel_map_X_mean24.ravel(), pixel_map_X_mean25.ravel(), pixel_map_X_mean26.ravel(),
                                    pixel_map_X_mean27.ravel(), pixel_map_X_mean28.ravel(), pixel_map_X_mean29.ravel(),
                                    pixel_map_X_mean30.ravel(),
                                    pixel_map_X_mean31.ravel(), pixel_map_X_mean32.ravel(), pixel_map_X_mean33.ravel(),
                                    pixel_map_X_mean34.ravel(), pixel_map_X_mean35.ravel(), pixel_map_X_mean36.ravel(),
                                    pixel_map_X_mean37.ravel(), pixel_map_X_mean38.ravel(), pixel_map_X_mean39.ravel(),
                                    pixel_map_X_mean40.ravel(),
                                    pixel_map_X_mean41.ravel(), pixel_map_X_mean42.ravel(), pixel_map_X_mean43.ravel(),
                                    pixel_map_X_mean44.ravel(), pixel_map_X_mean45.ravel(), pixel_map_X_mean46.ravel(),
                                    pixel_map_X_mean47.ravel(), pixel_map_X_mean48.ravel(), pixel_map_X_mean49.ravel(),
                                    pixel_map_X_mean50.ravel()), axis=0)

    camera_mean_y = np.concatenate((pixel_map_Y_mean1.ravel(), pixel_map_Y_mean2.ravel(), pixel_map_Y_mean3.ravel(),
                                    pixel_map_Y_mean4.ravel(), pixel_map_Y_mean5.ravel(), pixel_map_Y_mean6.ravel(),
                                    pixel_map_Y_mean7.ravel(), pixel_map_Y_mean8.ravel(), pixel_map_Y_mean9.ravel(),
                                    pixel_map_Y_mean10.ravel(),
                                    pixel_map_Y_mean11.ravel(), pixel_map_Y_mean12.ravel(), pixel_map_Y_mean13.ravel(),
                                    pixel_map_Y_mean14.ravel(), pixel_map_Y_mean15.ravel(), pixel_map_Y_mean16.ravel(),
                                    pixel_map_Y_mean17.ravel(), pixel_map_Y_mean18.ravel(), pixel_map_Y_mean19.ravel(),
                                    pixel_map_Y_mean20.ravel(),
                                    pixel_map_Y_mean21.ravel(), pixel_map_Y_mean22.ravel(), pixel_map_Y_mean23.ravel(),
                                    pixel_map_Y_mean24.ravel(), pixel_map_Y_mean25.ravel(), pixel_map_Y_mean26.ravel(),
                                    pixel_map_Y_mean27.ravel(), pixel_map_Y_mean28.ravel(), pixel_map_Y_mean29.ravel(),
                                    pixel_map_Y_mean30.ravel(),
                                    pixel_map_Y_mean31.ravel(), pixel_map_Y_mean32.ravel(), pixel_map_Y_mean33.ravel(),
                                    pixel_map_Y_mean34.ravel(), pixel_map_Y_mean35.ravel(), pixel_map_Y_mean36.ravel(),
                                    pixel_map_Y_mean37.ravel(), pixel_map_Y_mean38.ravel(), pixel_map_Y_mean39.ravel(),
                                    pixel_map_Y_mean40.ravel(),
                                    pixel_map_Y_mean41.ravel(), pixel_map_Y_mean42.ravel(), pixel_map_Y_mean43.ravel(),
                                    pixel_map_Y_mean44.ravel(), pixel_map_Y_mean45.ravel(), pixel_map_Y_mean46.ravel(),
                                    pixel_map_Y_mean47.ravel(), pixel_map_Y_mean48.ravel(), pixel_map_Y_mean49.ravel(),
                                    pixel_map_Y_mean50.ravel()), axis=0)

    camera_mean_z = np.concatenate((pixel_map_Z_mean1.ravel(), pixel_map_Z_mean2.ravel(), pixel_map_Z_mean3.ravel(),
                                    pixel_map_Z_mean4.ravel(), pixel_map_Z_mean5.ravel(), pixel_map_Z_mean6.ravel(),
                                    pixel_map_Z_mean7.ravel(), pixel_map_Z_mean8.ravel(), pixel_map_Z_mean9.ravel(),
                                    pixel_map_Z_mean10.ravel(),
                                    pixel_map_Z_mean11.ravel(), pixel_map_Z_mean12.ravel(), pixel_map_Z_mean13.ravel(),
                                    pixel_map_Z_mean14.ravel(), pixel_map_Z_mean15.ravel(), pixel_map_Z_mean16.ravel(),
                                    pixel_map_Z_mean17.ravel(), pixel_map_Z_mean18.ravel(), pixel_map_Z_mean19.ravel(),
                                    pixel_map_Z_mean20.ravel(),
                                    pixel_map_Z_mean21.ravel(), pixel_map_Z_mean22.ravel(), pixel_map_Z_mean23.ravel(),
                                    pixel_map_Z_mean24.ravel(), pixel_map_Z_mean25.ravel(), pixel_map_Z_mean26.ravel(),
                                    pixel_map_Z_mean27.ravel(), pixel_map_Z_mean28.ravel(), pixel_map_Z_mean29.ravel(),
                                    pixel_map_Z_mean30.ravel(),
                                    pixel_map_Z_mean31.ravel(), pixel_map_Z_mean32.ravel(), pixel_map_Z_mean33.ravel(),
                                    pixel_map_Z_mean34.ravel(), pixel_map_Z_mean35.ravel(), pixel_map_Z_mean36.ravel(),
                                    pixel_map_Z_mean37.ravel(), pixel_map_Z_mean38.ravel(), pixel_map_Z_mean39.ravel(),
                                    pixel_map_Z_mean40.ravel(),
                                    pixel_map_Z_mean41.ravel(), pixel_map_Z_mean42.ravel(), pixel_map_Z_mean43.ravel(),
                                    pixel_map_Z_mean44.ravel(), pixel_map_Z_mean45.ravel(), pixel_map_Z_mean46.ravel(),
                                    pixel_map_Z_mean47.ravel(), pixel_map_Z_mean48.ravel(), pixel_map_Z_mean49.ravel(),
                                    pixel_map_Z_mean50.ravel()), axis=0)


    # generate spatial measured mesh
    spatialmesh_x = np.linspace(63, -63, 8)
    spatialmesh_y = np.linspace(63, -63, 8)
    spatialmesh_z = np.linspace(200, 200 + (50 * 5), 50)

    spatialmesh_X, spatialmesh_Z, spatialmesh_Y = np.meshgrid(spatialmesh_x, spatialmesh_z, spatialmesh_y)
    spatialtrue_x = spatialmesh_X.ravel()
    spatialtrue_y = spatialmesh_Y.ravel()
    spatialtrue_z = spatialmesh_Z.ravel()

    # plt11 = plt.figure(figsize=(8, 6))
    # ax11 = plt.axes(projection='3d')
    # plt.title("Spatial True Points")
    # ax11.set_xlabel('X(t)')
    # ax11.set_ylabel('Z(t)')
    # ax11.set_zlabel('Y(t)')
    # ax11.scatter3D(spatialtrue_x, spatialtrue_y, spatialtrue_z, marker='x', color='C0')
    # plt11.show()

    # plt13 = plt.figure(figsize=(8, 6))
    # ax13 = plt.axes(projection='3d')
    # plt.title("Camera mean Points")
    # ax13.set_xlabel('X(t)')
    # ax13.set_ylabel('Z(t)')
    # ax13.set_zlabel('Y(t)')
    # ax13.scatter3D(spatialtrue_x, spatialtrue_y, spatialtrue_z, marker='o', color='C0')
    # ax13.scatter3D(camera_mean_x, camera_mean_y, camera_mean_z, marker='x', color='C1')
    # plt.legend(['Spatial', 'Camera'], framealpha=1)
    # plt13.show()
    #
    # plt141 = plt.figure(figsize=(15, 6))
    # plt.subplot(321)
    # plt.plot(camera_mean_x)
    # plt.subplot(322)
    # plt.plot(spatialtrue_x)
    # plt.subplot(323)
    # plt.plot(camera_mean_y)
    # plt.subplot(324)
    # plt.plot(spatialtrue_y)
    # plt.subplot(325)
    # plt.plot(camera_mean_z)
    # plt.subplot(326)
    # plt.plot(spatialtrue_z)
    # plt141.show()

    # plt11 = plt.figure(figsize=(8, 6))
    # ax11 = plt.axes(projection='3d')
    # plt.title("Spatial True Points")
    # ax11.set_xlabel('X(t)')
    # ax11.set_ylabel('Z(t)')
    # ax11.set_zlabel('Y(t)')
    # ax11.scatter3D(test_X_Dframe_mean_test3, test_Y_Dframe_mean_test3, test_Z_Dframe_mean_test3, marker='x', color='C0')
    # ax11.scatter3D(camera_mean_x, camera_mean_y, camera_mean_z, marker='.', color='C1')
    # plt11.show()

    return spatialtrue_x, spatialtrue_y, spatialtrue_z, camera_mean_x, camera_mean_y, camera_mean_z


spatialtrue_x, spatialtrue_y, spatialtrue_z, camera_mean_x, camera_mean_y, camera_mean_z = train_data()

t1_start = time.perf_counter()
INVQuadratic_coef,x_model, y_model, z_model = quadratic_regression_3d(spatialtrue_x, spatialtrue_y, spatialtrue_z, camera_mean_x, camera_mean_y,
                                            camera_mean_z)
t1_stop = time.perf_counter()
print("INVQuadratic Elapsed time:", (t1_stop - t1_start))

t2_start = time.perf_counter()
INVTPS_coef, INVx_TPS, INVy_TPS, INVz_TPS = thinplatespline(spatialtrue_x, spatialtrue_y, spatialtrue_z, camera_mean_x,
                                                            camera_mean_y, camera_mean_z)
t2_stop = time.perf_counter()
print("INV_TPS Elapsed time:", (t2_stop - t2_start))

plt21 = plt.figure(figsize=(8, 6))
ax21 = plt.axes(projection='3d')
# plt.title("INVQuadratic")
ax21.set_xlabel('X(t)')
ax21.set_ylabel('Z(t)')
ax21.set_zlabel('Y(t)')
ax21.scatter3D(spatialtrue_x, spatialtrue_y, spatialtrue_z, marker='o', color='C0')
ax21.scatter3D(x_model, y_model, z_model, marker='x', color='C1')
# plt.legend(['Spatial', 'MODEL'], framealpha=1)
plt21.show()

plt21 = plt.figure(figsize=(8, 6))
ax21 = plt.axes(projection='3d')
# plt.title("INV_TPS")
ax21.set_xlabel('X(t)')
ax21.set_ylabel('Z(t)')
ax21.set_zlabel('Y(t)')
ax21.scatter3D(spatialtrue_x, spatialtrue_y, spatialtrue_z, marker='o', color='C0')
ax21.scatter3D(INVx_TPS, INVy_TPS, INVz_TPS, marker='x', color='C1')
# plt.legend(['Spatial', 'MODEL'], framealpha=1)
plt21.show()

np.save("C:/Users/treen/PycharmProjects/pythonProject/EURASIP/BenchINVTPS_coef", INVTPS_coef)
np.save("C:/Users/treen/PycharmProjects/pythonProject/EURASIP/BenchINVQuadratic_coef", INVQuadratic_coef)