import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Cube:
    pts = np.zeros((54, 3), dtype='int')
    color_names = ['white', 'orange', 'blue', 'yellow', 'red', 'green']
    cs  = []

    def __init__(self):
        idx = 0    
        for x in [-2, 0, 2]:
            for y in [-2, 0, 2]:
                for s, c in enumerate(self.color_names):
                    z = (-1)**s * 3
                    self.pts[idx, (idx+0) % 3] = x
                    self.pts[idx, (idx+1) % 3] = y
                    self.pts[idx, (idx+2) % 3] = z
                    self.cs.append(c)
                    idx += 1

    def rotate_up(self, n):
        self.rotate_z(n, up=True)

    def rotate_down(self, n):
        self.rotate_z(n, up=False)
        
    def rotate_left(self, n):
        self.rotate_x(n, right=False)

    def rotate_right(self, n):
        self.rotate_x(n, right=True)      

    def rotate_front(self, n):
        self.rotate_y(n, front=True)

    def rotate_back(self, n):
        self.rotate_y(n, front=False)              
        
    def rotate_x(self, n, right):
        if right == True:
            idx_rot, = np.where(self.pts[:, 0] >=  2)
        else:
            idx_rot, = np.where(self.pts[:, 0] <= -2)
        
        theta = n * (np.pi / 2)
        M = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]],
                     dtype='int')
        fs_rot = np.dot(self.pts[idx_rot, 1:], M)
        self.pts[idx_rot, 1:] = fs_rot

        
    def rotate_y(self, n, front):
        if front == True:
            idx_rot, = np.where(self.pts[:, 1] <= -2)
        else:
            idx_rot, = np.where(self.pts[:, 1] >= 2)
            
        theta = n * (np.pi / 2)
        M = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]],
                     dtype='int')
        fs_rot = np.dot(self.pts[idx_rot, 0::2], M)
        self.pts[idx_rot, 0::2] = fs_rot

        
    def rotate_z(self, n, up):
        if up == True:
            idx_rot, = np.where(self.pts[:, 2] >=  2)
        else:
            idx_rot, = np.where(self.pts[:, 2] <= -2)
            
        theta = n * (np.pi / 2)
        M = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]],
                     dtype='int')
        fs_rot = np.dot(self.pts[idx_rot, 0:2], M)
        self.pts[idx_rot, 0:2] = fs_rot

        
    def cloud_plot(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.pts[:,0], self.pts[:,1], self.pts[:,2], c = self.cs,
                   edgecolors='k', s =100)
        ax.set_xlim([-3.01,3.01])
        ax.set_ylim([-3.01,3.01])
        ax.set_zlim([-3.01,3.01])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.axis('equal')
        return ax

if __name__ == "__main__":
    c = Cube()
    #c.rotate_front(1)
    # c.rotate_left(1)
    # c.rotate_up(2)
    # c.rotate_down(1)
    # c.rotate_right(1)
    # c.rotate_back(2)
    c.cloud_plot()
    plt.show()
