import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
        self.rotate_z(n, 'up')
    
    def rotate_down(self, n):
        self.rotate_z(n, 'down')
        
    def rotate_left(self, n):
        self.rotate_x(n, 'left')
    
    def rotate_right(self, n):
        self.rotate_x(n, 'right')      
    
    def rotate_front(self, n):
        self.rotate_y(n, 'front')
    
    def rotate_back(self, n):
        self.rotate_y(n, 'back')              
        
    def rotate_x(self, n, edge = None):
        if edge == 'right':
            idx_rot, = np.where(self.pts[:, 0] >=  2)
        elif edge == 'left':
            idx_rot, = np.where(self.pts[:, 0] <= -2)
        else:
            idx_rot = np.arange(0, 54)
            
        theta = n * (np.pi / 2)
        M = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]],
                     dtype='int')
        fs_rot = np.dot(self.pts[idx_rot, 1:], M)
        self.pts[idx_rot, 1:] = fs_rot
    
    def rotate_y(self, n, edge = None):
        if edge == 'front':
            idx_rot, = np.where(self.pts[:, 1] <= -2)
        elif edge == 'back':
            idx_rot, = np.where(self.pts[:, 1] >= 2)
        else:
            idx_rot = np.arange(0, 54)
            
        theta = n * (np.pi / 2)
        M = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]],
                     dtype='int')
        fs_rot = np.dot(self.pts[idx_rot, 0::2], M)
        self.pts[idx_rot, 0::2] = fs_rot
    
    def rotate_z(self, n, edge=None):
        if edge == 'up':
            idx_rot, = np.where(self.pts[:, 2] >=  2)
        elif edge == 'down':
            idx_rot, = np.where(self.pts[:, 2] <= -2)
        else:
            idx_rot = np.arange(0, 54)            
            
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
        ax.axis('off')
        plt.axis('equal')
        return ax

    def cube_plot(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, projection='3d')
        
        for idx, p in enumerate(self.pts):
            if abs(p[2]) > 2.5:
                xs = np.linspace(p[0]-1, p[0]+1, 2)
                ys = np.linspace(p[1]-1, p[1]+1, 2)                
                xs,ys = np.meshgrid(xs,ys)
                zs = p[2] * np.ones(np.shape(xs))

            if abs(p[0]) > 2.5:
                ys = np.linspace(p[1]-1, p[1]+1, 2)
                zs = np.linspace(p[2]-1, p[2]+1, 2)                
                ys,zs = np.meshgrid(ys,zs)
                xs = p[0] * np.ones(np.shape(zs))

            if abs(p[1]) > 2.5:
                xs = np.linspace(p[0]-1, p[0]+1, 2)
                zs = np.linspace(p[2]-1, p[2]+1, 2)                
                xs,zs = np.meshgrid(xs,zs)
                ys = p[1] * np.ones(np.shape(xs))
                                
            ax.plot_surface(xs,ys,zs, color = self.cs[idx],
                            edgecolor='k', shade=False, linewidth=3)
                
        ax.set_xlim([-3.01,3.01])
        ax.set_ylim([-3.01,3.01])
        ax.set_zlim([-3.01,3.01])
        ax.axis('off')
        plt.axis('equal')
        return ax

    
if __name__ == "__main__":
    c = Cube()
    #c.cloud_plot()
    c.cube_plot()
    c.rotate_z(-1)
    c.cube_plot()
    plt.show()
