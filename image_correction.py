##############################################
# THESE CODE IS USED TO
# CORRECT IMAGE DISTORTIONS
#
# E. tRENTO jR.

#import
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from scipy.spatial.distance import cdist
import time

def pixelcorrection(INVTPS_coef, x_true, y_true, z_true, x, y, z):
    #  THESE FUNCTION USES THE DISTORTION COEF AND CORRECT THE DISTORTED POINTS
    # training data
    # x_true,y_true,z_true
    # test data
    # x, y, z

    # tps coef
    wa_x = INVTPS_coef[0, :, 0]
    wa_y = INVTPS_coef[1, :, 0]
    wa_z = INVTPS_coef[2, :, 0]

    N = len(x_true)
    D = 3  # number of spatial dimensions

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

    # plot
    N_dense = len(x)
    pt_true = np.zeros((N, 3), dtype=float)
    pt_dist = np.zeros((N_dense, 3), dtype=float)

    pt_true[:, 0] = x_true
    pt_true[:, 1] = y_true
    pt_true[:, 2] = z_true

    pt_dist[:, 0] = x
    pt_dist[:, 1] = y
    pt_dist[:, 2] = z

    x_invTPS, y_invTPS, z_invTPS = f(pt_dist)

    return x_invTPS, y_invTPS, z_invTPS

path_train_mean_Dense = 'C:/Users/treen/PycharmProjects/pythonProject/dense_point_cloud/Bancada_otica/Dense/Medias/'
for i in range(1, 52):
    exec(f'pixel_map_X_mean{i}= np.load(path_train_mean_Dense + f"pixel_map_X_mean{i}.npy")')
    exec(f'pixel_map_Y_mean{i}= np.load(path_train_mean_Dense + f"pixel_map_Y_mean{i}.npy")')
    exec(f'pixel_map_Z_mean{i}= np.load(path_train_mean_Dense + f"pixel_map_Z_mean{i}.npy")')
    exec(f'pixel_map_X_mean_test{i}= np.load(path_train_mean_Dense + f"pixel_map_X_mean_test{i}.npy")')
    exec(f'pixel_map_Y_mean_test{i}= np.load(path_train_mean_Dense + f"pixel_map_Y_mean_test{i}.npy")')
    exec(f'pixel_map_Z_mean_test{i}= np.load(path_train_mean_Dense + f"pixel_map_Z_mean_test{i}.npy")')

path_test_Depth_frame_mean= 'C:/Users/treen/PycharmProjects/pythonProject/dense_point_cloud/Bancada_otica/Depth_frame/Medias/'
for i in range(1,4):
  exec(f"test_X_Dframe_mean{i}  = np.load(path_test_Depth_frame_mean  +'pixel_map_X_mean{i}.npy')")
  exec(f"test_Y_Dframe_mean{i}  = np.load(path_test_Depth_frame_mean  +'pixel_map_Y_mean{i}.npy')")
  exec(f"test_Z_Dframe_mean{i}  = np.load(path_test_Depth_frame_mean  +'pixel_map_Z_mean{i}.npy')")
  exec(f"test_X_Dframe_mean_test{i} = np.load(path_test_Depth_frame_mean  +'pixel_map_X_mean_test{i}.npy')")
  exec(f"test_Y_Dframe_mean_test{i} = np.load(path_test_Depth_frame_mean  +'pixel_map_Y_mean_test{i}.npy')")
  exec(f"test_Z_Dframe_mean_test{i} = np.load(path_test_Depth_frame_mean  +'pixel_map_Z_mean_test{i}.npy')")

  # Remove BaCkground
  exec(f"idx_background{i} = np.where((test_Z_Dframe_mean{i}.ravel() <= pixel_map_Z_mean{i}.min())|"
       f"(test_X_Dframe_mean{i}.ravel() >= pixel_map_X_mean{i}.max() ) | (test_X_Dframe_mean{i}.ravel() <= pixel_map_X_mean{i}.min())|"
       f"(test_Y_Dframe_mean{i}.ravel() >= pixel_map_Y_mean{i}.max() ) | (test_Y_Dframe_mean{i}.ravel() <= pixel_map_Y_mean{i}.min()))")
  exec(f"test_X_Dframe_mean{i} = np.delete(test_X_Dframe_mean{i}.ravel(), idx_background{i}, 0)")
  exec(f"test_Y_Dframe_mean{i} = np.delete(test_Y_Dframe_mean{i}.ravel(), idx_background{i}, 0)")
  exec(f"test_Z_Dframe_mean{i} = np.delete(test_Z_Dframe_mean{i}.ravel(), idx_background{i}, 0)")

  exec(f"idx_background_test{i} = np.where((test_Z_Dframe_mean_test{i}.ravel() <= pixel_map_Z_mean_test{i}.min())|"
       f"(test_X_Dframe_mean_test{i}.ravel() >= pixel_map_X_mean_test{i}.max() ) | (test_X_Dframe_mean_test{i}.ravel() <= pixel_map_X_mean_test{i}.min())|"
       f"(test_Y_Dframe_mean_test{i}.ravel() >= pixel_map_Y_mean_test{i}.max() ) | (test_Y_Dframe_mean_test{i}.ravel() <= pixel_map_Y_mean_test{i}.min()))")
  exec(f"test_X_Dframe_mean_test{i} = np.delete(test_X_Dframe_mean_test{i}.ravel(), idx_background_test{i}, 0)")
  exec(f"test_Y_Dframe_mean_test{i} = np.delete(test_Y_Dframe_mean_test{i}.ravel(), idx_background_test{i}, 0)")
  exec(f"test_Z_Dframe_mean_test{i} = np.delete(test_Z_Dframe_mean_test{i}.ravel(), idx_background_test{i}, 0)")


def train_data():
    #plot all train mean data
    camera_mean_x = np.concatenate((pixel_map_X_mean1.ravel(), pixel_map_X_mean2.ravel(), pixel_map_X_mean3.ravel(), pixel_map_X_mean4.ravel(), pixel_map_X_mean5.ravel(), pixel_map_X_mean6.ravel(), pixel_map_X_mean7.ravel(), pixel_map_X_mean8.ravel(), pixel_map_X_mean9.ravel(), pixel_map_X_mean10.ravel(),
                                  pixel_map_X_mean11.ravel(), pixel_map_X_mean12.ravel(), pixel_map_X_mean13.ravel(), pixel_map_X_mean14.ravel(), pixel_map_X_mean15.ravel(), pixel_map_X_mean16.ravel(), pixel_map_X_mean17.ravel(), pixel_map_X_mean18.ravel(), pixel_map_X_mean19.ravel(), pixel_map_X_mean20.ravel(),
                                  pixel_map_X_mean21.ravel(), pixel_map_X_mean22.ravel(), pixel_map_X_mean23.ravel(), pixel_map_X_mean24.ravel(), pixel_map_X_mean25.ravel(), pixel_map_X_mean26.ravel(), pixel_map_X_mean27.ravel(), pixel_map_X_mean28.ravel(), pixel_map_X_mean29.ravel(), pixel_map_X_mean30.ravel(),
                                  pixel_map_X_mean31.ravel(), pixel_map_X_mean32.ravel(), pixel_map_X_mean33.ravel(), pixel_map_X_mean34.ravel(), pixel_map_X_mean35.ravel(), pixel_map_X_mean36.ravel(), pixel_map_X_mean37.ravel(), pixel_map_X_mean38.ravel(), pixel_map_X_mean39.ravel(), pixel_map_X_mean40.ravel(),
                                  pixel_map_X_mean41.ravel(), pixel_map_X_mean42.ravel(), pixel_map_X_mean43.ravel(), pixel_map_X_mean44.ravel(), pixel_map_X_mean45.ravel(), pixel_map_X_mean46.ravel(), pixel_map_X_mean47.ravel(), pixel_map_X_mean48.ravel(), pixel_map_X_mean49.ravel(), pixel_map_X_mean50.ravel()), axis=0)

    camera_mean_y = np.concatenate((pixel_map_Y_mean1.ravel(), pixel_map_Y_mean2.ravel(), pixel_map_Y_mean3.ravel(), pixel_map_Y_mean4.ravel(), pixel_map_Y_mean5.ravel(), pixel_map_Y_mean6.ravel(), pixel_map_Y_mean7.ravel(), pixel_map_Y_mean8.ravel(), pixel_map_Y_mean9.ravel(), pixel_map_Y_mean10.ravel(),
                                  pixel_map_Y_mean11.ravel(), pixel_map_Y_mean12.ravel(), pixel_map_Y_mean13.ravel(), pixel_map_Y_mean14.ravel(), pixel_map_Y_mean15.ravel(), pixel_map_Y_mean16.ravel(), pixel_map_Y_mean17.ravel(), pixel_map_Y_mean18.ravel(), pixel_map_Y_mean19.ravel(), pixel_map_Y_mean20.ravel(),
                                  pixel_map_Y_mean21.ravel(), pixel_map_Y_mean22.ravel(), pixel_map_Y_mean23.ravel(), pixel_map_Y_mean24.ravel(), pixel_map_Y_mean25.ravel(), pixel_map_Y_mean26.ravel(), pixel_map_Y_mean27.ravel(), pixel_map_Y_mean28.ravel(), pixel_map_Y_mean29.ravel(), pixel_map_Y_mean30.ravel(),
                                  pixel_map_Y_mean31.ravel(), pixel_map_Y_mean32.ravel(), pixel_map_Y_mean33.ravel(), pixel_map_Y_mean34.ravel(), pixel_map_Y_mean35.ravel(), pixel_map_Y_mean36.ravel(), pixel_map_Y_mean37.ravel(), pixel_map_Y_mean38.ravel(), pixel_map_Y_mean39.ravel(), pixel_map_Y_mean40.ravel(),
                                  pixel_map_Y_mean41.ravel(), pixel_map_Y_mean42.ravel(), pixel_map_Y_mean43.ravel(), pixel_map_Y_mean44.ravel(), pixel_map_Y_mean45.ravel(), pixel_map_Y_mean46.ravel(), pixel_map_Y_mean47.ravel(), pixel_map_Y_mean48.ravel(), pixel_map_Y_mean49.ravel(), pixel_map_Y_mean50.ravel()), axis=0)

    camera_mean_z = np.concatenate((pixel_map_Z_mean1.ravel(), pixel_map_Z_mean2.ravel(), pixel_map_Z_mean3.ravel(), pixel_map_Z_mean4.ravel(), pixel_map_Z_mean5.ravel(), pixel_map_Z_mean6.ravel(), pixel_map_Z_mean7.ravel(), pixel_map_Z_mean8.ravel(), pixel_map_Z_mean9.ravel(), pixel_map_Z_mean10.ravel(),
                                  pixel_map_Z_mean11.ravel(), pixel_map_Z_mean12.ravel(), pixel_map_Z_mean13.ravel(), pixel_map_Z_mean14.ravel(), pixel_map_Z_mean15.ravel(), pixel_map_Z_mean16.ravel(), pixel_map_Z_mean17.ravel(), pixel_map_Z_mean18.ravel(), pixel_map_Z_mean19.ravel(), pixel_map_Z_mean20.ravel(),
                                  pixel_map_Z_mean21.ravel(), pixel_map_Z_mean22.ravel(), pixel_map_Z_mean23.ravel(), pixel_map_Z_mean24.ravel(), pixel_map_Z_mean25.ravel(), pixel_map_Z_mean26.ravel(), pixel_map_Z_mean27.ravel(), pixel_map_Z_mean28.ravel(), pixel_map_Z_mean29.ravel(), pixel_map_Z_mean30.ravel(),
                                  pixel_map_Z_mean31.ravel(), pixel_map_Z_mean32.ravel(), pixel_map_Z_mean33.ravel(), pixel_map_Z_mean34.ravel(), pixel_map_Z_mean35.ravel(), pixel_map_Z_mean36.ravel(), pixel_map_Z_mean37.ravel(), pixel_map_Z_mean38.ravel(), pixel_map_Z_mean39.ravel(), pixel_map_Z_mean40.ravel(),
                                  pixel_map_Z_mean41.ravel(), pixel_map_Z_mean42.ravel(), pixel_map_Z_mean43.ravel(), pixel_map_Z_mean44.ravel(), pixel_map_Z_mean45.ravel(), pixel_map_Z_mean46.ravel(), pixel_map_Z_mean47.ravel(), pixel_map_Z_mean48.ravel(), pixel_map_Z_mean49.ravel(), pixel_map_Z_mean50.ravel()), axis=0)

    # plt12 = plt.figure(figsize=(8, 6))
    # ax12 = plt.axes(projection='3d')
    # plt.title("Camera mean Points")
    # ax12.set_xlabel('X(t)')
    # ax12.set_ylabel('Z(t)')
    # ax12.set_zlabel('Y(t)')
    # ax12.scatter3D(camera_mean_x, camera_mean_y, camera_mean_z, marker='x', color='C1')
    # plt12.show()

    # generate spatial measured mesh
    spatialmesh_x = np.linspace(63, -63, 8)
    spatialmesh_y = np.linspace(63, -63, 8)
    spatialmesh_z = np.linspace(178.7, 178.7 + (50 * 5), 50)

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
    #
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

INVTPS_coef = np.load("C:/Users/treen/PycharmProjects/pythonProject/EURASIP/INVTPS_coef.npy")
spatialtrue_x, spatialtrue_y, spatialtrue_z, camera_mean_x, camera_mean_y, camera_mean_z = train_data()

t2_start = time.perf_counter()
x_Correct_frame_reto_mean, y_Correct_frame_reto_mean, z_Correct_frame_reto_mean = pixelcorrection(INVTPS_coef, camera_mean_x, camera_mean_y, camera_mean_z,
                                                                                   test_X_Dframe_mean_test3.ravel(), test_Y_Dframe_mean_test3.ravel(), test_Z_Dframe_mean_test3.ravel())
t2_stop = time.perf_counter()
print("Frame1 correction Elapsed time:", (t2_stop - t2_start))


plt22 = plt.figure(figsize=(8, 6))
ax22 = plt.axes(projection='3d')
plt.title("Corrected Reto Points")
ax22.set_xlabel('X(t)')
ax22.set_ylabel('Z(t)')
ax22.set_zlabel('Y(t)')
ax22.scatter3D(spatialtrue_x, spatialtrue_y, spatialtrue_z, marker='o', color='C0')
ax22.scatter3D(x_Correct_frame_reto_mean, y_Correct_frame_reto_mean, z_Correct_frame_reto_mean, marker='x', color='C1')
plt.legend(['Camera','Correct'], framealpha=1)
plt22.show()
