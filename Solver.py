import numpy as np
import matplotlib.pyplot as plt
from Cube import *


class Solver:
    def __init__(self, c):
        self.c = c

    def white_face_to_top(self):
        f = face_dict[c.center_piece_idx('white')[0]]
        self.c.rotate_face_to_face(f, 'up')

    def s1_remove_ep_from_right_edge(self, edge_to_switch):
        self.c.rotate_face_edge_to_edge('right', edge_to_switch, 'down')
        self.c.rotate_down(1)
        self.c.rotate_face_edge_to_edge('right', 'down', edge_to_switch)
        
    def solve_white_cross(self):
        # Ensure white cross on top
        self.white_face_to_top()
        self.c.cube_plot()

        for ep_to_solve in self.c.edge_pieces_matching_color('white'):
            adj_ep_to_solve = ep_to_solve.adj_pieces[0]
            
            if not c.edge_piece_solved(ep_to_solve):
                # use the adjacent piece to identify the target face color
                target_adj_face_color = adj_ep_to_solve.color_name
                target_face_piece = self.c.face_piece_matching_color(target_adj_face_color)

                if ep_to_solve.face_name in ['front', 'back', 'left', 'right']:
                    self.c.rotate_face_to_face(ep_to_solve.face_name, 'right')

                    # Target color is on the right, so we're already on the correct side
                    if self.c.face_color('right') == adj_ep_to_solve.color_name:
                        self.c.rotate_face_edge_to_edge('right', adj_ep_to_solve.face_name, 'front')
                        
                    # Target color is on another side, so move piece to correct side and flip to right
                    else:
                        self.s1_remove_ep_from_right_edge(adj_ep_to_solve.face_name)
                        self.c.rotate_face_edge_to_edge('down', 'front', target_face_piece.face_name)
                        self.c.rotate_face_to_face(target_face_piece.face_name, 'right')
                        self.c.rotate_face_edge_to_edge('right', 'down', 'front')

                    # the piece is now on the right face, front edge. solve
                    self.c.rotate_up(1)
                    self.c.rotate_front(-1)
                    self.c.rotate_up(-1)

                else:
                    self.c.rotate_face_to_face(adj_ep_to_solve.face_name, 'right')                    
                    if ep_to_solve.face_name == 'up':
                        self.c.rotate_right(2)
                    self.c.rotate_face_edge_to_edge('down', adj_ep_to_solve.face_name, target_face_piece.face_name)
                    self.c.rotate_face_to_face(target_face_piece.face_name, 'right')
                    self.c.rotate_right(2)
            
            
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
    
    
    # # Sample test config:
    # c.rotate_x(1)
    # c.rotate_up(-2)
    # c.rotate_left(1)
    # c.rotate_front(1)
    # c.rotate_down(1)
    # c.rotate_right(3)
    # c.rotate_up(1)
    # c.rotate_back(1)
    # c.rotate_down(1)
    # c.rotate_left(1)

    
    s = Solver(c)

    s.solve_white_cross()
    c.cube_plot()
    plt.show()
