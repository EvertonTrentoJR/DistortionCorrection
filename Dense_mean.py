##############################################
# THESE CODE IS USED TO CALCULATE THE MEAN POINT GRID
# FOR THE MODEL DISTORTION CONTROL GROUP
# AND SAVES AT THE FOLDER "MEDIAS"
#
# E. tRENTO jR.


import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

for n in range(1, 51):
    exec(
        f'path{n} = "C:/Users/treen/PycharmProjects/pythonProject/dense_point_cloud/setup_paper/tanque/Dense_tanque'
        f'/Test_{n}/"')

for n in range(1, 51):
    for i in range(1, 101):
        exec(f'pixel_map_X{n}_{i} = np.load(path{n} + "pixel_map_mX{i}.npy")')
        exec(f'pixel_map_Y{n}_{i} = np.load(path{n} + "pixel_map_mY{i}.npy")')
        exec(f'pixel_map_Z{n}_{i} = np.load(path{n} + "pixel_map_mZ{i}.npy")')

    exec(
        f'pixel_map_X_train{n} = np.dstack([pixel_map_X{n}_1, pixel_map_X{n}_2, pixel_map_X{n}_3, pixel_map_X{n}_4, '
        f'pixel_map_X{n}_5, pixel_map_X{n}_6, pixel_map_X{n}_7, pixel_map_X{n}_8, pixel_map_X{n}_9, pixel_map_X'
        f'{n}_10,pixel_map_X{n}_11, pixel_map_X{n}_12, pixel_map_X{n}_13, pixel_map_X{n}_14, pixel_map_X{n}_15, '
        f'pixel_map_X{n}_16, pixel_map_X{n}_17, pixel_map_X{n}_18, pixel_map_X{n}_19, pixel_map_X{n}_20,pixel_map_X'
        f'{n}_21, pixel_map_X{n}_22, pixel_map_X{n}_23, pixel_map_X{n}_24, pixel_map_X{n}_25, pixel_map_X{n}_26, '
        f'pixel_map_X{n}_27, pixel_map_X{n}_28, pixel_map_X{n}_29, pixel_map_X{n}_30,pixel_map_X{n}_31, pixel_map_X'
        f'{n}_32, pixel_map_X{n}_33, pixel_map_X{n}_34, pixel_map_X{n}_35, pixel_map_X{n}_36, pixel_map_X{n}_37, '
        f'pixel_map_X{n}_38, pixel_map_X{n}_39, pixel_map_X{n}_40,pixel_map_X{n}_41, pixel_map_X{n}_42, pixel_map_X'
        f'{n}_43, pixel_map_X{n}_44, pixel_map_X{n}_45, pixel_map_X{n}_46, pixel_map_X{n}_47, pixel_map_X{n}_48, '
        f'pixel_map_X{n}_49, pixel_map_X{n}_50, pixel_map_X{n}_51, pixel_map_X{n}_52, pixel_map_X{n}_53, pixel_map_X'
        f'{n}_54, pixel_map_X{n}_55, pixel_map_X{n}_56, pixel_map_X{n}_57, pixel_map_X{n}_58, pixel_map_X{n}_59, '
        f'pixel_map_X{n}_60,pixel_map_X{n}_61, pixel_map_X{n}_62, pixel_map_X{n}_63, pixel_map_X{n}_64, pixel_map_X'
        f'{n}_65, pixel_map_X{n}_66, pixel_map_X{n}_67, pixel_map_X{n}_68, pixel_map_X{n}_69, pixel_map_X{n}_70,'
        f'pixel_map_X{n}_71, pixel_map_X{n}_72, pixel_map_X{n}_73, pixel_map_X{n}_74, pixel_map_X{n}_75, pixel_map_X'
        f'{n}_76, pixel_map_X{n}_77, pixel_map_X{n}_78, pixel_map_X{n}_79, pixel_map_X{n}_80, pixel_map_X{n}_81, '
        f'pixel_map_X{n}_82, pixel_map_X{n}_83, pixel_map_X{n}_84, pixel_map_X{n}_85, pixel_map_X{n}_86, pixel_map_X'
        f'{n}_87, pixel_map_X{n}_88, pixel_map_X{n}_89, pixel_map_X{n}_90,pixel_map_X{n}_91, pixel_map_X{n}_92, '
        f'pixel_map_X{n}_93, pixel_map_X{n}_94, pixel_map_X{n}_95, pixel_map_X{n}_96, pixel_map_X{n}_97, pixel_map_X'
        f'{n}_98, pixel_map_X{n}_99, pixel_map_X{n}_100])')
    exec(
        f'pixel_map_Y_train{n} = np.dstack([pixel_map_Y{n}_1, pixel_map_Y{n}_2, pixel_map_Y{n}_3, pixel_map_Y{n}_4, '
        f'pixel_map_Y{n}_5, pixel_map_Y{n}_6, pixel_map_Y{n}_7, pixel_map_Y{n}_8, pixel_map_Y{n}_9, pixel_map_Y'
        f'{n}_10,pixel_map_Y{n}_11, pixel_map_Y{n}_12, pixel_map_Y{n}_13, pixel_map_Y{n}_14, pixel_map_Y{n}_15, '
        f'pixel_map_Y{n}_16, pixel_map_Y{n}_17, pixel_map_Y{n}_18, pixel_map_Y{n}_19, pixel_map_Y{n}_20,pixel_map_Y'
        f'{n}_21, pixel_map_Y{n}_22, pixel_map_Y{n}_23, pixel_map_Y{n}_24, pixel_map_Y{n}_25, pixel_map_Y{n}_26, '
        f'pixel_map_Y{n}_27, pixel_map_Y{n}_28, pixel_map_Y{n}_29, pixel_map_Y{n}_30,pixel_map_Y{n}_31, pixel_map_Y'
        f'{n}_32, pixel_map_Y{n}_33, pixel_map_Y{n}_34, pixel_map_Y{n}_35, pixel_map_Y{n}_36, pixel_map_Y{n}_37, '
        f'pixel_map_Y{n}_38, pixel_map_Y{n}_39, pixel_map_Y{n}_40,pixel_map_Y{n}_41, pixel_map_Y{n}_42, pixel_map_Y'
        f'{n}_43, pixel_map_Y{n}_44, pixel_map_Y{n}_45, pixel_map_Y{n}_46, pixel_map_Y{n}_47, pixel_map_Y{n}_48, '
        f'pixel_map_Y{n}_49, pixel_map_Y{n}_50, pixel_map_Y{n}_51, pixel_map_Y{n}_52, pixel_map_Y{n}_53, pixel_map_Y'
        f'{n}_54, pixel_map_Y{n}_55, pixel_map_Y{n}_56, pixel_map_Y{n}_57, pixel_map_Y{n}_58, pixel_map_Y{n}_59, '
        f'pixel_map_Y{n}_60, pixel_map_Y{n}_61, pixel_map_Y{n}_62, pixel_map_Y{n}_63, pixel_map_Y{n}_64, pixel_map_Y'
        f'{n}_65, pixel_map_Y{n}_66, pixel_map_Y{n}_67, pixel_map_Y{n}_68, pixel_map_Y{n}_69, pixel_map_Y{n}_70,'
        f'pixel_map_Y{n}_71, pixel_map_Y{n}_72, pixel_map_Y{n}_73, pixel_map_Y{n}_74, pixel_map_Y{n}_75, pixel_map_Y'
        f'{n}_76, pixel_map_Y{n}_77, pixel_map_Y{n}_78, pixel_map_Y{n}_79, pixel_map_Y{n}_80,pixel_map_Y{n}_81, '
        f'pixel_map_Y{n}_82, pixel_map_Y{n}_83, pixel_map_Y{n}_84, pixel_map_Y{n}_85, pixel_map_Y{n}_86, pixel_map_Y'
        f'{n}_87, pixel_map_Y{n}_88, pixel_map_Y{n}_89, pixel_map_Y{n}_90, pixel_map_Y{n}_91, pixel_map_Y{n}_92, '
        f'pixel_map_Y{n}_93, pixel_map_Y{n}_94, pixel_map_Y{n}_95, pixel_map_Y{n}_96, pixel_map_Y{n}_97, pixel_map_Y'
        f'{n}_98, pixel_map_Y{n}_99, pixel_map_Y{n}_100])')
    exec(
        f'pixel_map_Z_train{n} = np.dstack([pixel_map_Z{n}_1, pixel_map_Z{n}_2, pixel_map_Z{n}_3, pixel_map_Z{n}_4, '
        f'pixel_map_Z{n}_5, pixel_map_Z{n}_6, pixel_map_Z{n}_7, pixel_map_Z{n}_8, pixel_map_Z{n}_9, pixel_map_Z'
        f'{n}_10,pixel_map_Z{n}_11, pixel_map_Z{n}_12, pixel_map_Z{n}_13, pixel_map_Z{n}_14, pixel_map_Z{n}_15, '
        f'pixel_map_Z{n}_16, pixel_map_Z{n}_17, pixel_map_Z{n}_18, pixel_map_Z{n}_19, pixel_map_Z{n}_20,pixel_map_Z'
        f'{n}_21, pixel_map_Z{n}_22, pixel_map_Z{n}_23, pixel_map_Z{n}_24, pixel_map_Z{n}_25, pixel_map_Z{n}_26, '
        f'pixel_map_Z{n}_27, pixel_map_Z{n}_28, pixel_map_Z{n}_29, pixel_map_Z{n}_30,pixel_map_Z{n}_31, pixel_map_Z'
        f'{n}_32, pixel_map_Z{n}_33, pixel_map_Z{n}_34, pixel_map_Z{n}_35, pixel_map_Z{n}_36, pixel_map_Z{n}_37, '
        f'pixel_map_Z{n}_38, pixel_map_Z{n}_39, pixel_map_Z{n}_40,pixel_map_Z{n}_41, pixel_map_Z{n}_42, pixel_map_Z'
        f'{n}_43, pixel_map_Z{n}_44, pixel_map_Z{n}_45, pixel_map_Z{n}_46, pixel_map_Z{n}_47, pixel_map_Z{n}_48, '
        f'pixel_map_Z{n}_49, pixel_map_Z{n}_50, pixel_map_Z{n}_51, pixel_map_Z{n}_52, pixel_map_Z{n}_53, pixel_map_Z'
        f'{n}_54, pixel_map_Z{n}_55, pixel_map_Z{n}_56, pixel_map_Z{n}_57, pixel_map_Z{n}_58, pixel_map_Z{n}_59, '
        f'pixel_map_Z{n}_60,pixel_map_Z{n}_61, pixel_map_Z{n}_62, pixel_map_Z{n}_63, pixel_map_Z{n}_64, pixel_map_Z'
        f'{n}_65, pixel_map_Z{n}_66, pixel_map_Z{n}_67, pixel_map_Z{n}_68, pixel_map_Z{n}_69, pixel_map_Z{n}_70,'
        f'pixel_map_Z{n}_71, pixel_map_Z{n}_72, pixel_map_Z{n}_73, pixel_map_Z{n}_74, pixel_map_Z{n}_75, pixel_map_Z'
        f'{n}_76, pixel_map_Z{n}_77, pixel_map_Z{n}_78, pixel_map_Z{n}_79, pixel_map_Z{n}_80,pixel_map_Z{n}_81, '
        f'pixel_map_Z{n}_82, pixel_map_Z{n}_83, pixel_map_Z{n}_84, pixel_map_Z{n}_85, pixel_map_Z{n}_86, pixel_map_Z'
        f'{n}_87, pixel_map_Z{n}_88, pixel_map_Z{n}_89, pixel_map_Z{n}_90,pixel_map_Z{n}_91, pixel_map_Z{n}_92, '
        f'pixel_map_Z{n}_93, pixel_map_Z{n}_94, pixel_map_Z{n}_95, pixel_map_Z{n}_96, pixel_map_Z{n}_97, pixel_map_Z'
        f'{n}_98, pixel_map_Z{n}_99, pixel_map_Z{n}_100])')

for n in range(1, 51):
    exec(f'pixel_map_X_mean{n} = np.zeros([8,8])')
    exec(f'pixel_map_Y_mean{n} = np.zeros([8,8])')
    exec(f'pixel_map_Z_mean{n} = np.zeros([8,8])')

    for i in range(0, 8):
        for j in range(0, 8):
            exec(f'pixel_map_X_mean{n}[i,j] = np.mean(pixel_map_X_train{n}[i,j,:])')
            exec(f'pixel_map_Y_mean{n}[i,j] = np.mean(pixel_map_Y_train{n}[i,j,:])')
            exec(f'pixel_map_Z_mean{n}[i,j] = np.mean(pixel_map_Z_train{n}[i,j,:])')

# # plot all mean data
# camera_mean_x = np.concatenate((pixel_map_X_mean1.ravel(), pixel_map_X_mean2.ravel(), pixel_map_X_mean3.ravel(),
#                                 pixel_map_X_mean4.ravel(), pixel_map_X_mean5.ravel(), pixel_map_X_mean6.ravel(),
#                                 pixel_map_X_mean7.ravel(), pixel_map_X_mean8.ravel(), pixel_map_X_mean9.ravel(),
#                                 pixel_map_X_mean10.ravel(),
#                                 pixel_map_X_mean11.ravel(), pixel_map_X_mean12.ravel(), pixel_map_X_mean13.ravel(),
#                                 pixel_map_X_mean14.ravel(), pixel_map_X_mean15.ravel(), pixel_map_X_mean16.ravel(),
#                                 pixel_map_X_mean17.ravel(), pixel_map_X_mean18.ravel(), pixel_map_X_mean19.ravel(),
#                                 pixel_map_X_mean20.ravel(),
#                                 pixel_map_X_mean21.ravel(), pixel_map_X_mean22.ravel(), pixel_map_X_mean23.ravel(),
#                                 pixel_map_X_mean24.ravel(), pixel_map_X_mean25.ravel(), pixel_map_X_mean26.ravel(),
#                                 pixel_map_X_mean27.ravel(), pixel_map_X_mean28.ravel(), pixel_map_X_mean29.ravel(),
#                                 pixel_map_X_mean30.ravel(),
#                                 pixel_map_X_mean31.ravel(), pixel_map_X_mean32.ravel(), pixel_map_X_mean33.ravel(),
#                                 pixel_map_X_mean34.ravel(), pixel_map_X_mean35.ravel(), pixel_map_X_mean36.ravel(),
#                                 pixel_map_X_mean37.ravel(), pixel_map_X_mean38.ravel(), pixel_map_X_mean39.ravel(),
#                                 pixel_map_X_mean40.ravel(),
#                                 pixel_map_X_mean41.ravel(), pixel_map_X_mean42.ravel(), pixel_map_X_mean43.ravel(),
#                                 pixel_map_X_mean44.ravel(), pixel_map_X_mean45.ravel(), pixel_map_X_mean46.ravel(),
#                                 pixel_map_X_mean47.ravel(), pixel_map_X_mean48.ravel(), pixel_map_X_mean49.ravel(),
#                                 pixel_map_X_mean50.ravel()), axis=0)
#
# camera_mean_y = np.concatenate((pixel_map_Y_mean1.ravel(), pixel_map_Y_mean2.ravel(), pixel_map_Y_mean3.ravel(),
#                                 pixel_map_Y_mean4.ravel(), pixel_map_Y_mean5.ravel(), pixel_map_Y_mean6.ravel(),
#                                 pixel_map_Y_mean7.ravel(), pixel_map_Y_mean8.ravel(), pixel_map_Y_mean9.ravel(),
#                                 pixel_map_Y_mean10.ravel(),
#                                 pixel_map_Y_mean11.ravel(), pixel_map_Y_mean12.ravel(), pixel_map_Y_mean13.ravel(),
#                                 pixel_map_Y_mean14.ravel(), pixel_map_Y_mean15.ravel(), pixel_map_Y_mean16.ravel(),
#                                 pixel_map_Y_mean17.ravel(), pixel_map_Y_mean18.ravel(), pixel_map_Y_mean19.ravel(),
#                                 pixel_map_Y_mean20.ravel(),
#                                 pixel_map_Y_mean21.ravel(), pixel_map_Y_mean22.ravel(), pixel_map_Y_mean23.ravel(),
#                                 pixel_map_Y_mean24.ravel(), pixel_map_Y_mean25.ravel(), pixel_map_Y_mean26.ravel(),
#                                 pixel_map_Y_mean27.ravel(), pixel_map_Y_mean28.ravel(), pixel_map_Y_mean29.ravel(),
#                                 pixel_map_Y_mean30.ravel(),
#                                 pixel_map_Y_mean31.ravel(), pixel_map_Y_mean32.ravel(), pixel_map_Y_mean33.ravel(),
#                                 pixel_map_Y_mean34.ravel(), pixel_map_Y_mean35.ravel(), pixel_map_Y_mean36.ravel(),
#                                 pixel_map_Y_mean37.ravel(), pixel_map_Y_mean38.ravel(), pixel_map_Y_mean39.ravel(),
#                                 pixel_map_Y_mean40.ravel(),
#                                 pixel_map_Y_mean41.ravel(), pixel_map_Y_mean42.ravel(), pixel_map_Y_mean43.ravel(),
#                                 pixel_map_Y_mean44.ravel(), pixel_map_Y_mean45.ravel(), pixel_map_Y_mean46.ravel(),
#                                 pixel_map_Y_mean47.ravel(), pixel_map_Y_mean48.ravel(), pixel_map_Y_mean49.ravel(),
#                                 pixel_map_Y_mean50.ravel()), axis=0)
#
# camera_mean_z = np.concatenate((pixel_map_Z_mean1.ravel(), pixel_map_Z_mean2.ravel(), pixel_map_Z_mean3.ravel(),
#                                 pixel_map_Z_mean4.ravel(), pixel_map_Z_mean5.ravel(), pixel_map_Z_mean6.ravel(),
#                                 pixel_map_Z_mean7.ravel(), pixel_map_Z_mean8.ravel(), pixel_map_Z_mean9.ravel(),
#                                 pixel_map_Z_mean10.ravel(),
#                                 pixel_map_Z_mean11.ravel(), pixel_map_Z_mean12.ravel(), pixel_map_Z_mean13.ravel(),
#                                 pixel_map_Z_mean14.ravel(), pixel_map_Z_mean15.ravel(), pixel_map_Z_mean16.ravel(),
#                                 pixel_map_Z_mean17.ravel(), pixel_map_Z_mean18.ravel(), pixel_map_Z_mean19.ravel(),
#                                 pixel_map_Z_mean20.ravel(),
#                                 pixel_map_Z_mean21.ravel(), pixel_map_Z_mean22.ravel(), pixel_map_Z_mean23.ravel(),
#                                 pixel_map_Z_mean24.ravel(), pixel_map_Z_mean25.ravel(), pixel_map_Z_mean26.ravel(),
#                                 pixel_map_Z_mean27.ravel(), pixel_map_Z_mean28.ravel(), pixel_map_Z_mean29.ravel(),
#                                 pixel_map_Z_mean30.ravel(),
#                                 pixel_map_Z_mean31.ravel(), pixel_map_Z_mean32.ravel(), pixel_map_Z_mean33.ravel(),
#                                 pixel_map_Z_mean34.ravel(), pixel_map_Z_mean35.ravel(), pixel_map_Z_mean36.ravel(),
#                                 pixel_map_Z_mean37.ravel(), pixel_map_Z_mean38.ravel(), pixel_map_Z_mean39.ravel(),
#                                 pixel_map_Z_mean40.ravel(),
#                                 pixel_map_Z_mean41.ravel(), pixel_map_Z_mean42.ravel(), pixel_map_Z_mean43.ravel(),
#                                 pixel_map_Z_mean44.ravel(), pixel_map_Z_mean45.ravel(), pixel_map_Z_mean46.ravel(),
#                                 pixel_map_Z_mean47.ravel(), pixel_map_Z_mean48.ravel(), pixel_map_Z_mean49.ravel(),
#                                 pixel_map_Z_mean50.ravel()), axis=0)
#
# plt12 = plt.figure(figsize=(8, 6))
# ax12 = plt.axes(projection='3d')
# ax12.set_xlabel('X(t)')
# ax12.set_ylabel('Z(t)')
# ax12.set_zlabel('Y(t)')
# ax12.scatter3D(camera_mean_x, camera_mean_y, camera_mean_z, marker='o', color='C0')
# plt12.show()

plt12 = plt.figure(figsize=(8, 6))
ax12 = plt.axes(projection='3d')
ax12.set_xlabel('X(t)')
ax12.set_ylabel('Z(t)')
ax12.set_zlabel('Y(t)')
ax12.scatter3D(pixel_map_X_mean26, pixel_map_Y_mean26, pixel_map_Z_mean26, marker='o', color='C0')
plt12.show()

# np.save
for i in range(1, 51):
    exec(
        f'np.save("C:/Users/treen/PycharmProjects/pythonProject/\dense_point_cloud/setup_paper/tanque/medias/" + "Dense_pixel_map_X_mean{i}", pixel_map_X_mean{i})')
    exec(
        f'np.save("C:/Users/treen/PycharmProjects/pythonProject/dense_point_cloud/setup_paper/tanque/medias/" + "Dense_pixel_map_Y_mean{i}", pixel_map_Y_mean{i})')
    exec(
        f'np.save("C:/Users/treen/PycharmProjects/pythonProject/dense_point_cloud/setup_paper/tanque/medias/" + "Dense_pixel_map_Z_mean{i}", pixel_map_Z_mean{i})')
