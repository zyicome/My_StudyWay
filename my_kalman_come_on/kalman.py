import numpy as np


class Kalman:
    def __init__(self, dt, X):
        self.F = np.array([[1., 0, 0, 0, dt, 0, 0, 0],
                           [0, 1., 0, 0, 0, dt, 0, 0],
                           [0, 0, 1., 0, 0, 0, dt, 0],
                           [0, 0, 0, 1., 0, 0, 0, dt],
                           [0, 0, 0, 0, 1., 0, 0, 0],
                           [0, 0, 0, 0, 0, 1., 0, 0],
                           [0, 0, 0, 0, 0, 0, 1., 0],
                           [0, 0, 0, 0, 0, 0, 0, 1.]])

        self.H = np.array([[1., 0, 0, 0, 0, 0, 0, 0],
                           [0, 1., 0, 0, 0, 0, 0, 0],
                           [0, 0, 1., 0, 0, 0, 0, 0],
                           [0, 0, 0, 1., 0, 0, 0, 0],
                           [0, 0, 0, 0, 1., 0, 0, 0],
                           [0, 0, 0, 0, 0, 1., 0, 0],
                           [0, 0, 0, 0, 0, 0, 1., 0],
                           [0, 0, 0, 0, 0, 0, 0, 1.]])

        self.R = np.array([[1., 0, 0, 0, 0, 0, 0, 0],
                           [0, 1., 0, 0, 0, 0, 0, 0],
                           [0, 0, 10., 0, 0, 0, 0, 0],
                           [0, 0, 0, 10., 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0]])

        self.P = np.array([[1., 0, 0, 0, 0, 0, 0, 0],
                           [0, 1., 0, 0, 0, 0, 0, 0],
                           [0, 0, 1., 0, 0, 0, 0, 0],
                           [0, 0, 0, 1., 0, 0, 0, 0],
                           [0, 0, 0, 0, 1., 0, 0, 0],
                           [0, 0, 0, 0, 0, 1., 0, 0],
                           [0, 0, 0, 0, 0, 0, 1., 0],
                           [0, 0, 0, 0, 0, 0, 0, 1.]])

        self.last_P = np.array([[1., 0, 0, 0, 0, 0, 0, 0],
                           [0, 1., 0, 0, 0, 0, 0, 0],
                           [0, 0, 1., 0, 0, 0, 0, 0],
                           [0, 0, 0, 1., 0, 0, 0, 0],
                           [0, 0, 0, 0, 1., 0, 0, 0],
                           [0, 0, 0, 0, 0, 1., 0, 0],
                           [0, 0, 0, 0, 0, 0, 1., 0],
                           [0, 0, 0, 0, 0, 0, 0, 1.]])

        self.Z = np.ones(1*4)

        self.W = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

        self.X = X

        self.last_X = X

        self.dt = dt

        self._std_weight_position = 1. / 20

        self._std_weight_velocity = 1. / 160

    def predict(self):
        std_pos = np.array([[
            self._std_weight_position,
            self._std_weight_position,
            1e-2,
            self._std_weight_position]]).astype(float)
        print("std_pos", std_pos)
        std_vel = np.array([[
            self._std_weight_velocity,
            self._std_weight_velocity,
            1e-5,
            self._std_weight_velocity]]).astype(float)
        print("std_vel", std_vel)
        a = np.r_[std_pos, std_vel]
        print(a)
        Q = np.diag(np.square(np.c_[std_pos, std_vel]))
        print("Q:", Q)
        self.X = np.dot(self.F, self.last_X)
        print(self.W.T)
        self.P = np.dot(np.dot(self.F, self.last_P), self.F.T) + Q


    def update(self, now_Z):
        std = [
            self._std_weight_position * self.X[3][0],
            self._std_weight_position * self.X[3][0],
            1e-1,
            self._std_weight_position * self.X[3][0],
            self._std_weight_position * self.X[3][0],
            self._std_weight_position * self.X[3][0],
            1e-1,
            self._std_weight_position * self.X[3][0]]
        R = np.diag(np.square(std))
        now_Z = now_Z.T
        K_process = np.dot(np.dot(self.H, self.P), self.H.T) + R
        print("K_process", K_process)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(K_process))
        print("K:", K)
        X_process = now_Z - np.dot(self.H, self.X)
        print("now_z：", now_Z.T)
        print("X_process:", X_process)
        X_process[4][0] = self.X[4][0]
        X_process[5][0] = self.X[5][0]
        X_process[6][0] = self.X[6][0]
        X_process[7][0] = self.X[7][0]
        self.X = self.X + np.dot(K, X_process)
        self.P = self.P - np.dot(K, self.H, self.P)
        Vx = (self.X[0][0] - self.last_X[0][0]) / self.dt
        Vy = (self.X[1][0] - self.last_X[1][0]) / self.dt
        Va = (self.X[2][0] - self.last_X[2][0]) / self.dt
        Vh = (self.X[3][0] - self.last_X[3][0]) / self.dt
        self.X[4] = Vx
        self.X[5] = Vy
        self.X[6] = Va
        self.X[7] = Vh
        self.last_X = self.X
        self.last_P = self.P

    def show_matrix(self):
        print("X:", self.X)
        print("P:", self.P)
        print("H:", self.H)
        print("F:", self.F)
        print("R:", self.R)
        print("Z:", self.Z)
        print("W:", self.W)


