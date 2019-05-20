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

opposite_face = {
    'front' : 'back',  'left' : 'right', 'up' : 'down',
    'back'  : 'front', 'right' : 'left', 'down' : 'up',
    face_dict['front'] : face_dict['back'],
    face_dict['back']  : face_dict['front'],    
    face_dict['right'] : face_dict['left'],
    face_dict['left']  : face_dict['right'],    
    face_dict['up'] : face_dict['down'],
    face_dict['down']  : face_dict['up']    
}

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

    def edge(self):
        if self.piece_type != 'edge':
            raise Exception("Not an edge piece!")        
        return set([self.face_name, self.adj_pieces[0].face_name])
        
    def corner(self):
        if self.piece_type != 'corner':
            raise Exception("Not a corner piece!")        
        return set([self.face_name, self.adj_pieces[0].face_name, self.adj_pieces[1].face_name])

    def adjacent_color_names(self):
        return set([p.color_name for p in self.adj_pieces])
            
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
                    idx = self.point_to_idx(self.pts[ii,:])                    

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

    def point_to_idx(self, pt):
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

    def idx_to_point(self, idx):
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

    def corner_pieces(self, corner):
        return [cp for cp in self.pieces if cp.piece_type == 'corner' and cp.corner() == corner]

    def idx_to_corner(self, idx):
        if self.pieces_cube[idx].piece_type == 'corner':
            return self.pieces_cube[idx].corner()           

    def pt_to_corner(self, pt):
        idx = self.point_to_idx(pt)
        return self.idx_to_corner(idx)
    
    def update_after_rotation(self, ii_rot):
        for ii in ii_rot:
            idx = self.point_to_idx(self.pts[ii, :])
            self.pieces[ii].update_indices(self.pts[ii, :], idx)
            self.pieces_cube[idx]  = self.pieces[ii]

    def rotate_cube_face_to_face(self, f1, f2):
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
        # same faces, so nothing to do:
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
                rot_ax = face_normals[f]
                n_rot  = 2

            # they're distinct faces, and we just need a single rotation
            else:
                rot_ax = rot_ax / np.linalg.norm(rot_ax)
                rot_ax = rot_ax.astype(int)                
                n_rot = -np.dot(rot_ax.flatten(), face_normals[f])

            self.rotate_face(f, n_rot)

                
    # def rotate_cube_corner_to_corner(self, c1, c2):
    #     # we will keep track of where c1 goes using one of its pieces
    #     c1_pieces = self.corner_pieces(c1)
      
    #     # we may mutate these below
    #     c1 = list(c1)
    #     c2 = list(c2)
    #     c1.sort()
    #     c2.sort()
    #     c1_pieces = [p for p,_ in sorted(zip(c1_pieces, c1), key = lambda pair:pair[1])]
        
    #     if c1 == c2:
    #         pass
    #     else:
    #         # list of common faces
    #         f_common = list(set(c1).intersection(set(c2)))
    #         f_common.sort()

    #         # if any faces are in common, use the first
    #         if f_common:
    #             f_common = f_common[0]

    #         # no faces in common, so every face of c1 is opposite c2
    #         # so we'll rotate the cube to flip the first face of c2 with its opposite
    #         # (which corresponds to one of c1's faces
    #         else:
    #             f_common = c2[0]
    #             self.rotate_cube_face_to_face(opposite_face[f_common], f_common)
    #             c1 = list(c1_pieces[0].corner())
    #             c1.sort()

    #         # # Now we have c1 and c2 on the face f_common, we'll use the same
    #         # # steps as the face corner-to-corner algorithm from here. 
    #         # c1.remove(f_common)
    #         # c2.remove(f_common)
            
    #         # v1 = face_normals[c1[0]] + face_normals[c1[1]]
    #         # v2 = face_normals[c2[0]] + face_normals[c2[1]]
    #         # rot_ax = np.cross(v1, v2)

    #         # # v1 and v2 are not parallel, so it's a +/- 90* turn
    #         # if np.linalg.norm(rot_ax) > 0:
    #         #     rot_ax = rot_ax / np.linalg.norm(rot_ax)
    #         #     rot_ax = rot_ax.astype(int)
    #         #     if max(rot_ax) > 0:
    #         #         n_rot  = 1
    #         #     else:
    #         #         n_rot = -1
                    
    #         # # v1 and v2 are parallel, so it's a 180* turn
    #         # else:
    #         #     n_rot = 2
    #         #     rot_ax = face_normals[f_common]
            
    #         # # perform rotation
    #         # if   abs(rot_ax[2]) == 1:
    #         #     self.rotate_z(n_rot)
    #         # elif abs(rot_ax[0]) == 1:
    #         #     self.rotate_x(n_rot)
    #         # elif abs(rot_ax[1]) == 1:
    #         #     self.rotate_y(-n_rot)

    def rotate_face_corner_to_corner(self, f, c1, c2):
        c1 = list(c1)
        c2 = list(c2)
        
        # same faces, so nothing to do:
        if c1 == c2:
            pass

        # faces to switch:
        else :
            c1.remove(f)
            c2.remove(f)
            
            v1 = face_normals[c1[0]] + face_normals[c1[1]]
            v2 = face_normals[c2[0]] + face_normals[c2[1]]
            rot_ax = np.cross(v1, v2)

            # v1 and v2 are not parallel, so it's a +/- 90* turn
            if np.linalg.norm(rot_ax) > 0:
                rot_ax = rot_ax / np.linalg.norm(rot_ax)
                rot_ax = rot_ax.astype(int)
                if max(rot_ax) > 0:
                    n_rot  = 1
                else:
                    n_rot =  -1
                    
            # v1 and v2 are parallel, so it's a 180* turn
            else:
                n_rot = 2
                rot_ax = face_normals[f]
            
            # perform rotation
            f = face_dict[f]
            if   abs(rot_ax[2]) == 1:
                self.rotate_z(n_rot, edge=f)
            elif abs(rot_ax[0]) == 1:
                self.rotate_x(n_rot, edge=f)
            elif abs(rot_ax[1]) == 1:
                self.rotate_y(-n_rot, edge=f)

    def rotate_face(self, face_name, n):
        if face_name == 'up':
            self.rotate_up(n)
            
        if face_name == 'down':
            self.rotate_down(n)
            
        if face_name == 'left':
            self.rotate_left(n)
            
        if face_name == 'right':
            self.rotate_right(n)
            
        if face_name == 'front':
            self.rotate_front(n)
            
        if face_name == 'back':
            self.rotate_back(n)
    
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
            n = -n
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
            n = -n
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
            n = -n
        else:
            ii_rot = np.arange(0, 54)            
            
        theta = n * (np.pi / 2)
        M = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]],
                     dtype='int')
        fs_rot = np.dot(self.pts[ii_rot, 0:2], M)
        self.pts[ii_rot, 0:2] = fs_rot
        self.update_after_rotation(ii_rot)

    def face_color(self, face_name):
        face = face_dict[face_name]
        return self.pieces_cube[face, 1, 1].color_name
        
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

    def corner_piece_solved(self, pc):
        if pc.piece_type != 'corner':
            raise Exception("Bad input to corner_piece_solved... not a corner piece!")
        
        adj_pcs = pc.adj_pieces
        
        idx = pc.idx
        idx_ctr = (idx[0], 1, 1)
        color = pc.color_name
        color_ctr = self.pieces_cube[idx_ctr].color_name
       
        adj_pc1  = adj_pcs[0]
        idx_adj1 = adj_pc1.idx        
        idx_adj1_ctr = (idx_adj1[0], 1, 1)
        color_adj1 = adj_pc1.color_name
        color_adj1_ctr = self.pieces_cube[idx_adj1_ctr].color_name        

        adj_pc2  = adj_pcs[1]
        idx_adj2 = adj_pc2.idx
        idx_adj2_ctr = (idx_adj2[0], 1, 1)
        color_adj2 = adj_pc2.color_name
        color_adj2_ctr = self.pieces_cube[idx_adj2_ctr].color_name        
        
        return (color, color_adj1, color_adj2) == (color_ctr, color_adj1_ctr, color_adj2_ctr)

    
    def edge_pieces_matching_color(self, c):
        return [pc for pc in self.pieces if
                pc.idx in idx_edge_pieces and (pc.color_name == c or pc.color_name == color_dict[c])]

    def center_piece_matching_color(self, c):
        return [pc for pc in self.pieces if
                pc.idx in idx_center_pieces and (pc.color_name == c or pc.color_name == color_dict[c])][0]

    def corner_pieces_matching_color(self, c):
        return [pc for pc in self.pieces if
                pc.idx in idx_corner_pieces and (pc.color_name == c or pc.color_name == color_dict[c])]
    
        
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

    ## Sample test config 2:
    #c.rotate_right(2)
    # c.rotate_left(2)
    # c.rotate_down(1)
    # c.rotate_right(2)
    # c.rotate_up(1)
    # c.rotate_right(1)
    # c.rotate_front(1)
    # c.rotate_down(1)
    # c.rotate_x(1)
    # c.rotate_x(-1)

    #c.cube_plot()
    
    # idx1 = c.point_to_idx(np.array([-2, -3, 2], dtype=int))
    # idx2 = c.point_to_idx(np.array([ 2,  2, 3], dtype=int))    
    # cp1 = c.pieces_cube[idx2]
    # cp2 = c.pieces_cube[idx1]
    # print(cp1.corner(), cp2.corner())

    # c.rotate_cube_corner_to_corner(cp1.corner(), cp2.corner())
    
    c.cube_plot()
    #c.square_plot()
    cp = c.pieces_cube[idx_corner_pieces[0]]
    print(cp.pt, c.corner_piece_solved(cp))
    plt.show()
