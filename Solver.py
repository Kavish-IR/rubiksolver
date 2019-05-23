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

                if not self.c.edge_piece_solved(ep_to_solve):
                    self.c.cube_plot('Something went wrong with the current edge piece.')
            
            
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
                tmp = sorted(list(piece_to_solve.corner()))
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
            if 'down' in piece_to_solve.corner():
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

            if not self.c.corner_piece_solved(piece_to_solve):                
                self.c.cube_plot(title_str = 'Something went wrong with the current corner piece.  {0}'.format(i))
                
            #self.c.cube_plot(title_str="{0} post solve".format(i))            
            # rotate the cube for the next edge to solve
            self.c.rotate_z(1)

    def remove_middle_edge_piece(self, e):
        self.c.rotate_cube_face_to_face(e, 'front')
        self.c.rotate_up(1)
        self.c.rotate_right(1)
        self.c.rotate_up(-1)
        self.c.rotate_right(-1)
        self.c.rotate_up(-1)
        self.c.rotate_front(-1)
        self.c.rotate_up(1)
        self.c.rotate_front(1)
        self.c.rotate_cube_face_to_face('front', e)        
            
    def solve_middle_edges(self):
        self.c.rotate_x(2)

        # We will solve corners one-by-one clock-wise
        for i in range(4):
            colors_to_solve = [self.c.face_color('front'), self.c.face_color('right')]
            ep_to_solve = self.c.edge_piece_matching_colors(colors_to_solve[0], colors_to_solve[1])

            if 'up' not in ep_to_solve.edge():
                if ep_to_solve.edge() == set(['right', 'back']):
                    self.remove_middle_edge_piece('right')
                elif ep_to_solve.edge() == set(['left', 'back']):
                    self.remove_middle_edge_piece('back')
                elif ep_to_solve.edge() == set(['left', 'front']):
                    self.remove_middle_edge_piece('left')
                elif ep_to_solve.edge() == set(['front', 'right']):
                    self.remove_middle_edge_piece('front')
                    
            if ep_to_solve.face_name == 'up':
                ep_to_solve = ep_to_solve.adj_pieces[0]
            starting_edge = ep_to_solve.face_name

            if ep_to_solve.color_name == self.c.face_color('front'):
                self.c.rotate_face_edge_to_edge('up', starting_edge, 'front')
                self.c.rotate_up(1)
                self.c.rotate_right(1)
                self.c.rotate_up(-1)
                self.c.rotate_right(-1)
                self.c.rotate_up(-1)
                self.c.rotate_front(-1)
                self.c.rotate_up(1)
                self.c.rotate_front(1)
                
            elif ep_to_solve.color_name == self.c.face_color('right'):
                self.c.rotate_face_edge_to_edge('up', starting_edge, 'right')
                self.c.rotate_cube_face_to_face('right', 'front')
                self.c.rotate_up(-1)
                self.c.rotate_left(-1)
                self.c.rotate_up(1)
                self.c.rotate_left(1)
                self.c.rotate_up(1)
                self.c.rotate_front(1)
                self.c.rotate_up(-1)
                self.c.rotate_front(-1)
                self.c.rotate_cube_face_to_face('front', 'right')
                
            self.c.rotate_cube_face_to_face('right', 'front')
    
    def solve_yellow_cross(self):
        yellow_edge_pieces = self.c.edge_pieces_matching_color('yellow')

        while len([ep for ep in yellow_edge_pieces if ep.face_name == 'up']) < 4:
            yellow_cross_pieces = [ep for ep in yellow_edge_pieces if ep.face_name == 'up']
            n_yellow_cross_in_place = len(yellow_cross_pieces)

            # None placed, state 2 from manual
            if n_yellow_cross_in_place == 0:
                # solve cross w/ State 2 method
                self.c.rotate_front(1)
                self.c.rotate_up(1)
                self.c.rotate_right(1)
                self.c.rotate_up(-1)
                self.c.rotate_right(-1)
                self.c.rotate_front(-1)

            else:                
                ep_adj1 = yellow_cross_pieces[0].adj_pieces[0]
                ep_adj2 = yellow_cross_pieces[1].adj_pieces[0]
                
                n1 = face_normals[ep_adj1.face_name]
                n2 = face_normals[ep_adj2.face_name]
                orientation = np.dot(np.cross(n1, n2), np.array([0,0,1], dtype='int'))
                
                # Two up yellow edge pieces are in a line, state 4 from manual, after rotation
                if orientation == 0:
                    # Line is perpendicular to front in this case. Needs to be parallel for State 4.
                    if ep_adj1.face_name in ['front', 'back']:
                        self.c.rotate_face_edge_to_edge('up', 'front', 'right')

                    # solve cross w/ State 4 method
                    self.c.rotate_front(1)
                    self.c.rotate_right(1)
                    self.c.rotate_up(1)
                    self.c.rotate_right(-1)
                    self.c.rotate_up(-1)
                    self.c.rotate_front(-1)

                # Two up yellow edge pieces are not in a line, state 3 from manual, after rotation.
                else:
                    # If n1,n2 is ccw oriented, then we can rotate ep_adj1's face to back to get state 3
                    if orientation > 0:
                        self.c.rotate_face_edge_to_edge('up', ep_adj1.face_name, 'back')
                    # Otherwise, we should rotate ep_adj2's face to back to get state 3.
                    if orientation < 0:
                        self.c.rotate_face_edge_to_edge('up', ep_adj2.face_name, 'back')

                    # solve cross w/  State 3 method
                    self.c.rotate_front(1)
                    self.c.rotate_up(1)
                    self.c.rotate_right(1)
                    self.c.rotate_up(-1)
                    self.c.rotate_right(-1)
                    self.c.rotate_front(-1)

    def non_up_yellow_corner_orientation(self, cp):
        n1 = face_normals[cp.face_name]
        n2 = [face_normals[cp_adj.face_name] for cp_adj in cp.adj_pieces if cp_adj.face_name != 'up'][0]
        orientation = np.dot( np.cross(n1, n2), face_normals['up'])
        return orientation
              
    def solve_yellow_corners(self):
        while len([ep for ep in self.c.corner_pieces_matching_color('yellow') if ep.face_name == 'up']) < 4:
            yellow_corner_pieces = self.c.corner_pieces_matching_color('yellow')            
            up_yellow_corner_pieces = [cp for cp in yellow_corner_pieces if cp.face_name == 'up']
            n_up_yellow_corners = len(up_yellow_corner_pieces)

            other_yellow_corner_pieces = [cp for cp in yellow_corner_pieces if cp not in up_yellow_corner_pieces]
            other_yellow_corner_orientations = [self.non_up_yellow_corner_orientation(cp) for cp in
                                                other_yellow_corner_pieces]
            
            # State 1. We need to have a yellow piece on the up left edge. Find a corner piece
            # on the up edge that we can rotate to put yellow on the left face. Do the rotation.

            if n_up_yellow_corners == 0:
                cp_to_rotate =  [cp for cp, orientation in
                                 list(zip(other_yellow_corner_pieces, other_yellow_corner_orientations))
                                 if orientation > 0][0]
                self.c.rotate_face_edge_to_edge('up', cp_to_rotate.face_name, 'left')

            # State 2. We have exactly 1 yellow piece with its face on the up face. It needs to be
            # in the up,left,front position. Do the rotation.
            if n_up_yellow_corners == 1:
                cp_to_rotate = [cp for cp in up_yellow_corner_pieces][0]
                adj_faces = set([cp.face_name for cp in cp_to_rotate.adj_pieces])

                # Move desired piece to front,left,up position
                if adj_faces == set(['left', 'back']):
                    self.c.rotate_face_edge_to_edge('up', 'left', 'front')
                elif adj_faces == set(['back', 'right']):
                    self.c.rotate_face_edge_to_edge('up', 'back', 'front')
                elif adj_faces == set(['front', 'right']):
                    self.c.rotate_face_edge_to_edge('up', 'right', 'front')
                else:
                    pass
            
            # State 3. We need to have a yellow piece on the up front edge. Find a corner piece
            # on the up edge that we can rotate to put yellow on the front face. Do the rotaiton.
            if n_up_yellow_corners >= 2:
                cp_to_rotate =  [cp for cp, orientation in
                                 list(zip(other_yellow_corner_pieces, other_yellow_corner_orientations))
                                 if orientation < 0][0]
                self.c.rotate_face_edge_to_edge('up', cp_to_rotate.face_name, 'front')

            # In position. Apply Stage 6 algorithm.
            self.c.rotate_right(1)
            self.c.rotate_up(1)
            self.c.rotate_right(-1)
            self.c.rotate_up(1)
            self.c.rotate_right(1)
            self.c.rotate_up(2)
            self.c.rotate_right(-1)
            
    
if __name__ == '__main__':
    c = Cube()

    ## Sample test config 2:
    c.rotate_right(2)
    c.rotate_left(2)
    c.rotate_down(1)
    c.rotate_right(2)
    c.rotate_up(1)
    c.rotate_right(1)
    c.rotate_front(1)
    c.rotate_down(1)
    c.rotate_x(1)
    c.rotate_up(2)
    c.rotate_right(2)
    
    
    #Sample test config:
    c.rotate_x(1)
    c.rotate_up(-2)
    c.rotate_left(1)
    #c.rotate_front(1)
    c.rotate_down(1)
    #c.rotate_right(3)
    c.rotate_up(1)
    c.rotate_back(1)
    c.rotate_down(1)
    c.rotate_left(1)

    s = Solver(c)

    c.cube_plot()
    s.solve_white_cross()
    c.cube_plot(title_str = 'after white cross')
    s.solve_white_corners()
    c.cube_plot(title_str = 'after white corners')
    s.solve_middle_edges()
    c.cube_plot(title_str = 'after middle edges')
    s.solve_yellow_cross()    
    c.cube_plot(title_str = 'after yellow cross')
    s.solve_yellow_corners()
    c.cube_plot(title_str = 'after yellow corners')    
    plt.show()
