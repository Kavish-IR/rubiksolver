import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle

color_dict = {0 : 'white',  1 : 'orange', 2 : 'blue',
              3 : 'yellow', 4 : 'red',    5 : 'green',
              'white'  : 0, 'orange' : 1, 'blue' :  2,
              'yellow' : 3, 'red'    : 4, 'green' : 5}
    
face_dict = {0 : 'front', 1 : 'right', 2 : 'back',
             3 : 'left',  4 : 'up',    5 : 'down',
             'front' : 0, 'right' : 1, 'back' : 2,
             'left'  : 3, 'up'    : 4, 'down' : 5} 

face_normals = {'front' : np.array([ 0, -1,  0], dtype=int),
                'back'  : np.array([ 0,  1,  0], dtype=int),
                'left'  : np.array([-1,  0,  0], dtype=int),
                'right' : np.array([ 1,  0,  0], dtype=int),
                'up'    : np.array([ 0,  0,  1], dtype=int),
                'down'  : np.array([ 0,  0, -1], dtype=int)}

color_names = ['white', 'orange', 'blue', 'yellow', 'red', 'green']

idx_edge_pieces = [(f,i_f,j_f) for f in range(6)
                   for i_f in range(3)
                   for j_f in range(3)
                   if (i_f == 1 or j_f == 1) and i_f != j_f]

idx_corner_pieces = [(f, i_f, j_f) for f in range(6)
                     for i_f in range(3)
                     for j_f in range(3)
                     if (i_f != 1 and j_f != 1)]

idx_center_pieces = [(f, 1, 1) for f in range(6)]


class Piece:
    def __init__(self, color_name, piece_type, ii, pt, idx, adj_pieces = []):
        self.color_name = color_name
        self.color      = color_dict[color_name]
        self.piece_type = piece_type
        self.ii = ii        
        self.pt = pt
        self.idx = idx
        self.face      = idx[0]
        self.face_name = face_dict[self.face]
        self.adj_pieces = adj_pieces

    def update_indices(self, pt, idx):
        self.pt = pt
        self.idx = idx
        self.face = idx[0]
        self.face_name = face_dict[self.face]
            
class Cube:
    pts = np.zeros((54, 3), dtype='int')
    pieces  = np.zeros(54, dtype=Piece)
    pieces_cube = np.zeros((6, 3, 3), dtype=Piece)
    cs  = []
    
    def __init__(self):
        ii = 0    
        for x in [-2, 0, 2]:
            for y in [-2, 0, 2]:
                for s, c in enumerate(color_names):
                    z = (-1)**s * 3

                    # Fill point array
                    self.pts[ii, (ii+0) % 3] = x
                    self.pts[ii, (ii+1) % 3] = y
                    self.pts[ii, (ii+2) % 3] = z
                    self.cs.append(c)

                    # Fill square array
                    idx = self.point_to_square_idx(self.pts[ii,:])                    

                    # Get piece type
                    if idx in idx_center_pieces:
                        piece_type = 'center'
                    elif idx in idx_edge_pieces:
                        piece_type = 'edge'
                    elif idx in idx_corner_pieces:
                        piece_type = 'corner'
                    
                    # Fill piece array
                    self.pieces[ii] = Piece(c, piece_type, ii, np.copy(self.pts[ii,:]), idx)
                    self.pieces_cube[idx] = self.pieces[ii]
                    
                    # Advance index
                    ii += 1
        self._init_adjacency_()

    # adjacent pieces need to know that they're adjacent, so we let them know:
    def _init_adjacency_(self):
        # The difference between any adjacent points lies in this list:
        stencil = np.array([[ 1,  1,  0], [ 1, -1,  0], [-1,  1,  0], [-1, -1,  0],
                            [ 0,  1,  1], [ 0,  1, -1], [ 0, -1,  1], [ 0, -1, -1],
                            [ 1,  0,  1], [ 1,  0, -1], [ -1, 0,  1], [ -1, 0, -1]],                            
                            dtype='int')

        # We'll use this below since numpy arrays are fickle to compare to
        pts_as_array_list = [list(x) for x in list(self.pts)]

        for pc in self.pieces:
            pt = pc.pt
            
            # any adjacent piece would have to be at one of these points:
            candidate_adj_list = list(pt + stencil)
            
            # but any adjacent point has to be on the cube!
            adj_pts = [p for p in candidate_adj_list if list(p) in pts_as_array_list]

            # now identify their list indices and associated pieces            
            adj_ii     = [pts_as_array_list.index(list(p)) for p in adj_pts]
            adj_pieces = [self.pieces[ii] for ii in adj_ii]
            pc.adj_pieces = adj_pieces

    def point_to_square_idx(self, pt):
        x,y,z = pt
        if   z ==  3:
            return face_dict['up'], int((2+x) / 2), int((2+y) / 2)
        elif z == -3:
            return face_dict['down'], int((2+x) / 2), 2 - int((2+y) / 2)
        elif x ==  3:
            return face_dict['right'], int((2+y) / 2), int((2+z) / 2)
        elif x == -3:
            return face_dict['left'], 2 - int((2+y) / 2), int((2+z) / 2)
        elif y ==  3:
            return face_dict['back'], 2 - int((2+x) / 2), int((2+z) / 2)
        elif y == -3:
            return face_dict['front'], int((2+x) / 2), int((2+z) / 2)

    def square_idx_to_point(self, idx):
        f,i,j = idx
        if   face_dict[f] == 'up':
            return np.array([2*i-2,  2*j-2, 3], dtype='int')
        elif face_dict[f] == 'down':
            return np.array([2*i-2,  2-2*j, -3], dtype='int')
        elif face_dict[f] == 'right':
            return np.array([3, 2*i-2, 2*j-2], dtype='int')            
        elif face_dict[f] == 'left':
            return np.array([-3, 2-2*i, 2*j-2], dtype='int')
        elif face_dict[f] == 'back':
            return np.array([2-2*i,  3, 2*j-2], dtype='int')
        elif face_dict[f] == 'front':
            return np.array([2*i-2, -3, 2*j-2], dtype='int')                    

    def face_of_idx(self, idx):
        return idx[0]

    #def face_of_color(self, color_name):
    #    face_p
    
    def edge_of_idx(self, idx):
        pass
       
    def update_after_rotation(self, ii_rot):
        for ii in ii_rot:
            idx = self.point_to_square_idx(self.pts[ii, :])
            self.pieces[ii].update_indices(self.pts[ii, :], idx)
            self.pieces_cube[idx]  = self.pieces[ii]

    def rotate_face_to_face(self, f1, f2):
        # same faces, so nohting to do:
        if f1 == f2:
            pass

        # faces to switch:
        else :
            n1 = face_normals[f1]
            n2 = face_normals[f2]
            rot_ax = np.cross(n1, n2)

            # Zero rot_ax implies n1 and n2 are opposite faces
            # and means we need to determine the rotation axis
            if int(np.linalg.norm(rot_ax)) == 0:

                # If n1 is parallel to x or y, we rotate 180 about z
                if abs(n1[0]) == 1 or abs(n1[1]) == 1:
                    rot_ax = face_normals['up']
                    n_rot  = 2

                # If n1 is parallel to z, we rotate 180 about x
                if abs(n1[2]) == 1:
                    rot_ax = face_normals['right']
                    n_rot  = 2

            # they're distinct faces, and we just need a single rotation
            else:
                if min(rot_ax) < 0:
                    n_rot = 1
                else:
                    n_rot = -1
        
            # perform rotation
            if   abs(rot_ax[2]) == 1:
                self.rotate_z(n_rot)
            elif abs(rot_ax[0]) == 1:
                self.rotate_x(n_rot)
            elif abs(rot_ax[1]) == 1:
                self.rotate_y(-n_rot)


    def rotate_face_edge_to_edge(self, f, e1, e2):
        # same faces, so nohting to do:
        if e1 == e2:
            pass

        # faces to switch:
        else :
            n1 = face_normals[e1]
            n2 = face_normals[e2]
            rot_ax = np.cross(n1, n2)

            # Zero rot_ax implies n1 and n2 are opposite faces
            # and means we need to determine the rotation axis
            if int(np.linalg.norm(rot_ax)) == 0:

                # If n1 is parallel to x or y, we rotate 180 about z
                if abs(n1[0]) == 1 or abs(n1[1]) == 1:
                    rot_ax = face_normals['up']
                    n_rot  = 2

                # If n1 is parallel to z, we rotate 180 about x
                if abs(n1[2]) == 1:
                    rot_ax = face_normals['right']
                    n_rot  = 2

            # they're distinct faces, and we just need a single rotation
            else:
                if min(rot_ax) < 0:
                    n_rot = 1
                else:
                    n_rot = -1
        
            # perform rotation
            f = face_dict[f]
            if   abs(rot_ax[2]) == 1:
                self.rotate_z(n_rot, edge=f)
            elif abs(rot_ax[0]) == 1:
                self.rotate_x(n_rot, edge=f)
            elif abs(rot_ax[1]) == 1:
                self.rotate_y(-n_rot, edge=f)
                
            
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
        if edge in ['right', face_dict['right']]:
            ii_rot, = np.where(self.pts[:, 0] >=  2)
        elif edge in ['left', face_dict['left']]:
            ii_rot, = np.where(self.pts[:, 0] <= -2)
        else:
            ii_rot = np.arange(0, 54)
            
        theta = n * (np.pi / 2)
        M = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]],
                     dtype='int')
        fs_rot = np.dot(self.pts[ii_rot, 1:], M)
        self.pts[ii_rot, 1:] = fs_rot
        self.update_after_rotation(ii_rot)
 
    
    def rotate_y(self, n, edge = None):
        if edge in ['front', face_dict['front']]:
            ii_rot, = np.where(self.pts[:, 1] <= -2)
        elif edge in ['back', face_dict['back']]:
            ii_rot, = np.where(self.pts[:, 1] >= 2)
        else:
            ii_rot = np.arange(0, 54)
            
        theta = n * (np.pi / 2)
        M = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]],
                     dtype='int')
        fs_rot = np.dot(self.pts[ii_rot, 0::2], M)
        self.pts[ii_rot, 0::2] = fs_rot
        self.update_after_rotation(ii_rot)
    
    def rotate_z(self, n, edge=None):
        if edge in ['up', face_dict['up']]:
            ii_rot, = np.where(self.pts[:, 2] >=  2)
        elif edge in ['down', face_dict['down']]:
            ii_rot, = np.where(self.pts[:, 2] <= -2)
        else:
            ii_rot = np.arange(0, 54)            
            
        theta = n * (np.pi / 2)
        M = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]],
                     dtype='int')
        fs_rot = np.dot(self.pts[ii_rot, 0:2], M)
        self.pts[ii_rot, 0:2] = fs_rot
        self.update_after_rotation(ii_rot)

    def face_color(self, face):
        face = face_dict[face]
        return self.pieces_cube[face_dict['right'], 1, 1].color_name
        
    def center_piece_idx(self, color_name):
        idx_list = [idx for idx in idx_center_pieces if self.pieces_cube[idx].color_name == color_name]
        return idx_list[0]        
        
    def edge_piece_idx(self, c):
        return [idx for idx in idx_edge_pieces if self.pieces_cube[idx].color_name == color_name]
    
    def corner_piece_idx(self, c):
        return [idx for idx in idx_corner_pieces if self.pieces_cube[idx].color_name == color_name]
                    
    def edge_piece_solved(self, pc):
        if pc.piece_type != 'edge':
            raise Exception("Bad input to edge_piece_solved... not an edge piece!")
        adj_pc = pc.adj_pieces[0]
        
        idx = pc.idx
        idx_ctr = (idx[0], 1, 1)
        idx_adj = adj_pc.idx
        idx_adj_ctr = (idx_adj[0], 1, 1)

        color = pc.color_name
        color_ctr = self.pieces_cube[idx_ctr].color_name
        color_adj = adj_pc.color_name
        color_adj_ctr = self.pieces_cube[idx_adj_ctr].color_name
        
        return color == color_ctr and color_adj == color_adj_ctr
    
    def edge_pieces_matching_color(self, c):
        return [pc for pc in self.pieces if
                pc.idx in idx_edge_pieces and (pc.color_name == c or pc.color_name == color_dict[c])]

    def face_piece_matching_color(self, c):
        return [pc for pc in self.pieces if
                pc.idx in idx_center_pieces and (pc.color_name == c or pc.color_name == color_dict[c])][0]

        
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

    def cube_plot(self, ax=None, title_str=""):
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
        ax.set_title(title_str)
        plt.axis('equal')
        return ax

    def square_plot(self):
        fig, axs = plt.subplots(2,3)        
        
        for f in range(6):
            i = f % 3
            j = int(f / 3)
            ax = axs[j][i]
            
            for s in range(9):
                i_f = s % 3
                j_f = int(s / 3)
                x = i_f
                y = j_f
                c = self.pieces_cube[f, i_f, j_f].color_name
                ax.add_patch(Rectangle((x, y), width = 1, height = 1,
                                       facecolor = c, linewidth=2,
                                       edgecolor='k'))
                ax.set_title(face_dict[f] + " " + str(f))
                ax.set_xlim(0, 3)
                ax.set_ylim(0, 3)                
                ax.axis('equal')
                ax.axis('off')
                
        plt.tight_layout()
        return axs
    
if __name__ == "__main__":
    c = Cube()

    idx = idx_edge_pieces[0]
    print(c.piece_cube[idx].color_name,
          c.piece_cube[idx].ii,
          c.piece_cube[idx].piece_type,
          c.piece_cube[idx].idx,
          c.piece_cube[idx].face_name,
          c.piece_cube[idx].pt)
    # ii = c.piece_cube[idx].ii
    
    # c.cube_plot()
    #c.cube_plot()
    #c.square_plot()
    #c.rotate_face_to_face('back', 'up')
    # # # c.cube_plot()
    #c.rotate_face_to_face('up', 'back')    
    # c.rotate_y(1)
    # c.cube_plot()

    #    idx = c.pieces[ii].idx
    
    # print(c.piece_cube[idx].color_name,
    #       c.piece_cube[idx].ii,          
    #       c.piece_cube[idx].piece_type,
    #       c.piece_cube[idx].idx,
    #       c.piece_cube[idx].pt)
    
    #c.cube_plot()
    #c.square_plot()
    #plt.show()
    pass
