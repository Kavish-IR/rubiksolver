import numpy as np
import matplotlib.pyplot as plt
from Cube import *


class Solver:
    def __init__(self, c):
        self.c = c

    def white_face_to_top(self):
        f = face_dict[self.c.center_piece_idx('white')[0]]
        self.c.rotate_cube_face_to_face(f, 'up')

    def remove_ep_from_right_edge(self, edge_to_switch):
        self.c.rotate_face_edge_to_edge('right', edge_to_switch, 'down')
        self.c.rotate_down(-1)
        self.c.rotate_face_edge_to_edge('right', 'down', edge_to_switch)

        
    def solve_white_cross(self):
        # Ensure white cross on top
        self.white_face_to_top()

        for ep_to_solve in self.c.edge_pieces_matching_color('white'):
            adj_ep_to_solve = ep_to_solve.adj_pieces[0]

            if not self.c.edge_piece_solved(ep_to_solve):                
                # use the adjacent piece to identify the target face color
                target_adj_face_color = adj_ep_to_solve.color_name
                target_face_piece = self.c.center_piece_matching_color(target_adj_face_color)

                if ep_to_solve.face_name in ['front', 'back', 'left', 'right']:
                    self.c.rotate_cube_face_to_face(ep_to_solve.face_name, 'right')

                    # Target color is on the right, so we're already on the correct side
                    if self.c.face_color('right') == adj_ep_to_solve.color_name:
                        self.c.rotate_face_edge_to_edge('right', adj_ep_to_solve.face_name, 'front')
                        
                    # Target color is on another side, so move piece to correct side and flip to right
                    else:
                        self.remove_ep_from_right_edge(adj_ep_to_solve.face_name)
                        self.c.rotate_face_edge_to_edge('down', 'front', target_face_piece.face_name)
                        self.c.rotate_cube_face_to_face(target_face_piece.face_name, 'right')
                        self.c.rotate_face_edge_to_edge('right', 'down', 'front')

                    # the piece is now on the right face, front edge. solve
                    self.c.rotate_up(1)
                    self.c.rotate_front(-1)
                    self.c.rotate_up(-1)

                else:
                    self.c.rotate_cube_face_to_face(adj_ep_to_solve.face_name, 'right')                    
                    if ep_to_solve.face_name == 'up':
                        self.c.rotate_right(2)
                    self.c.rotate_face_edge_to_edge('down', adj_ep_to_solve.face_name, target_face_piece.face_name)
                    self.c.rotate_cube_face_to_face(target_face_piece.face_name, 'right')
                    self.c.rotate_right(2)
            
            
    def solve_white_corners(self):        
        # White corner pieces:
        white_corner_pieces = self.c.corner_pieces_matching_color('white')
        
        # We will solve corners one-by-one clock-wise
        for i in range(4):
            pt_corner_to_solve = np.array([2, -2, 3], dtype='int')
            colors_to_solve = [c.face_color('right'), c.face_color('front')]

            # identify the corner we need to send to the target_corner
            for piece_to_solve in white_corner_pieces:
                if piece_to_solve.adjacent_color_names() == set(colors_to_solve):
                    break
                
            # all points will move here
            target_corner = set(['front', 'right', 'up'])

            # piece is unsolved, but in the target corner. this will move it to front,down,right position
            if piece_to_solve.corner() == target_corner:
                self.c.rotate_right(-1)
                self.c.rotate_down(-1)
                self.c.rotate_right(1)
                self.c.rotate_down(1)

            # piece is unsolved but in some other up corner. move it to front,down,right position
            elif 'up' in piece_to_solve.corner():
                tmp = list(piece_to_solve.corner())
                tmp.remove('up')
                if tmp[0] != 'right':                    
                    face_to_rotate = tmp[0]
                else:
                    face_to_rotate = tmp[1]

                corner_start = piece_to_solve.corner()
                corner_end   = tmp
                corner_end.append('down')
                corner_end   = set(corner_end)

                self.c.rotate_face_corner_to_corner(face_to_rotate, corner_start, corner_end)
                self.c.rotate_face_corner_to_corner('down', corner_end, set(['front', 'right', 'down']))
                self.c.rotate_face_corner_to_corner(face_to_rotate, corner_end, corner_start)

            # piece is on down edge. move to front,right,down position
            elif 'down' in piece_to_solve.corner():
                self.c.rotate_face_corner_to_corner('down', piece_to_solve.corner(), set(['front', 'right', 'down'])) 

            #self.c.cube_plot(title_str="{0} prior to solve".format(i))
            # case that white is on front:
            if piece_to_solve.face_name == 'front':
                self.c.rotate_down(-1)
                self.c.rotate_right(-1)
                self.c.rotate_down(1)
                self.c.rotate_right(1)

            # case that white is on right
            elif piece_to_solve.face_name == 'right':
                self.c.rotate_right(-1)
                self.c.rotate_down(-1)
                self.c.rotate_right(1)

            # case that white is on down face
            else:
                self.c.rotate_right(-1)
                self.c.rotate_down(-2)
                self.c.rotate_right(1)
                self.c.rotate_down(1)
                self.c.rotate_right(-1)
                self.c.rotate_down(-1)
                self.c.rotate_right(1)
            
            self.c.cube_plot(title_str="{0} post solve".format(i))
            
            # rotate the cube for the next edge to solve
            self.c.rotate_z(1)
                    
if __name__ == '__main__':
    c = Cube()

    # ## Sample test config 2:
    # c.rotate_right(2)
    # c.rotate_left(2)
    # c.rotate_down(1)
    # c.rotate_right(2)
    # c.rotate_up(1)
    # c.rotate_right(1)
    # c.rotate_front(1)
    # c.rotate_down(1)
    # c.rotate_x(1)
    # c.rotate_up(2)
    # c.rotate_right(2)
    
    
    #Sample test config:
    c.rotate_x(1)
    c.rotate_up(-2)
    c.rotate_left(1)
    c.rotate_front(1)
    c.rotate_down(1)
    c.rotate_right(3)
    c.rotate_up(1)
    c.rotate_back(1)
    c.rotate_down(1)
    c.rotate_left(1)

    
    s = Solver(c)

    c.cube_plot()
    s.solve_white_cross()
    c.cube_plot()
    s.solve_white_corners()
    plt.show()
