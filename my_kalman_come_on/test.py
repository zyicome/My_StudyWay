import kalman
import numpy as np

X = np.array([[100], [50], [1.2], [150], [0], [0], [0], [0]])
my_test_kalman = kalman.Kalman(0.02, X)
my_test_kalman.show_matrix()
print("--------------1-----------")
my_test_kalman.predict()
my_test_kalman.show_matrix()
now_Z = np.array([[136, 100, 1.5, 130, 0, 0, 0, 0]])
print("---------------update--------------")
my_test_kalman.update(now_Z)
my_test_kalman.show_matrix()
print("--------------2-----------")
my_test_kalman.predict()
my_test_kalman.show_matrix()
now_Z_2 = np.array([[182, 146, 1.7, 100, 0, 0, 0, 0]])
print("---------------update--------------")
my_test_kalman.update(now_Z_2)
my_test_kalman.show_matrix()
print("--------------3-----------")
my_test_kalman.predict()
my_test_kalman.show_matrix()
now_Z_3 = np.array([[200, 180, 2.2, 70, 0, 0, 0, 0]])
print("---------------update--------------")
my_test_kalman.update(now_Z_3)
my_test_kalman.show_matrix()
print("--------------4-----------")
my_test_kalman.predict()
my_test_kalman.show_matrix()
now_Z_4 = np.array([[230, 220, 2.6, 50, 0, 0, 0, 0]])
print("---------------update--------------")
my_test_kalman.update(now_Z_4)
my_test_kalman.show_matrix()
print("--------------5-----------")
my_test_kalman.predict()
my_test_kalman.show_matrix()
now_Z_5 = np.array([[265, 250, 3.1, 40, 0, 0, 0, 0]])
print("---------------update--------------")
my_test_kalman.update(now_Z_5)
my_test_kalman.show_matrix()
print("--------------5-----------")
my_test_kalman.predict()
my_test_kalman.show_matrix()

