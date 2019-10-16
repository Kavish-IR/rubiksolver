from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib import lines
import numpy as np
import sys
import cv2 as cv

class ManualSquareExtractor:
    def __init__(self, img, fig = None, ax = None):
        # set up figure for interface
        if fig is None or ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(111)

        # Set up canvas and image / etc
        self.fig = fig
        self.ax  = ax
        self.img = img
        self.c = self.fig.canvas
        self.show_img()

        # Figure elements 
        self.lines = []
        self.scatters = []

        # State for interface
        self.state = "add_squares"
        self.complete = False

        # Connect event listeners
        self.connect()

        # Set up storage for squares and faces
        self.square_list = []
        self.pts = np.zeros((4, 2))
        self.update_rightmost_point(np.zeros((1,2)))        
        self.faces = None
        
    def connect(self):
        # Add event listeners
        self.clicker  = self.c.mpl_connect("button_press_event",  lambda e: self.button_press(e))
        self.presser  = self.c.mpl_connect("key_press_event",     lambda e: self.key_press(e))
        self.follower = self.c.mpl_connect("motion_notify_event", lambda e: self.followmouse(e))
        
    def show_img(self):
        self.ax.imshow(self.img)
        self.ax.set_title('Click to add squares along NW to SE diagonal.')
        self.ax.set_aspect('equal')
        self.ax.set_xticks([],[])
        self.ax.set_yticks([],[])
        self.fig.canvas.draw()                
        
    def key_press(self, event):
        if event.key == 'ctrl+z':
            if len(self.square_list) == 3:
                self.show_img()
                self.update_state("add_squares")
            
            if len(self.square_list) > 0:
                self.remove_last_square()
                
        if event.key == 'q':
            self.complete = True

    def remove_last_square(self):
        # remove last square from list of squares
        self.square_list.pop()
        
        # update rightmost point, set to upper left corner if no squares remain
        if len(self.square_list) > 0:
            self.update_rightmost_point(self.square_list[-1][2, :].reshape((1,2)))
        else:
            self.update_rightmost_point(np.zeros((1,2)))

        # remove the last square from drawing
        self.undraw_last_square()
                
    def button_press(self, event):
        self.add_square(event)
                
    def undraw_last_square(self):
        l = self.lines.pop()
        s = self.scatters.pop()
        l.remove()
        s.remove()
        self.c.draw()
            
    def update_state(self, new_state):        
        if self.state in "add_squares":
            self.state = new_state
            self.c.mpl_disconnect(self.clicker)            
            self.follower = self.c.mpl_connect("motion_notify_event", self.followmouse)
            self.clicker = self.c.mpl_connect ("button_press_event",  self.releaseonclick)

        elif self.state in ["follow", "done"]:
            self.state = new_state
            self.c.mpl_disconnect(self.follower)
            self.c.mpl_disconnect(self.clicker)

            if new_state == "add_squares":
                self.clicker = self.c.mpl_connect("button_press_event", self.button_press)

    def update_rightmost_point(self, pt):
        self.rightmost_pt = np.copy(pt)        

    def n_squares(self):
        return len(self.square_list)
        
    def add_square(self, event):
        if (event.xdata <= self.rightmost_pt[0, 0] or
            event.ydata <= self.rightmost_pt[0, 1]):
            print("Can't release here. Release down / right of : {0}".format(self.rightmost_pt))
        else:
            # empty points and sccatter plot
            line,  = self.ax.plot([], [], c = "lime")
            scatter = self.ax.scatter([], [], c = "lime")

            # add scatters
            self.lines.append(line)
            self.scatters.append(scatter)
            
            self.pts = np.zeros((4,2))
            
            self.pts[:, 0] = event.xdata
            self.pts[:, 1] = event.ydata
            
            #self.rightmost_pt = np.copy(self.pts[2, :]).reshape((1,2))
            self.update_rightmost_point(self.pts[2, :].reshape((1,2)))
            
            # plot the points and lines
            self.update_lines()        
            
            # update the internal state
            self.update_state(new_state = 'follow')

    def update_square(self, p2):
        p1 = self.pts[0,:]
        self.pts[2, :] = p2

        # Set up vectors parallel to diagonals
        v = (p2 - p1).flatten()
        v_perp = np.array([v[1], -v[0]])

        # Set up vectors parallel to edges
        w = 0.5 * (v + v_perp)
        w_perp = np.array([w[1], -w[0]])
        if np.dot(v, w_perp) < 0:
            w_perp = -w_perp
            
        # update points
        self.pts[1, :] = p1 + w
        self.pts[3, :] = p1 + w_perp
        
    def update_lines(self):
        if len(self.lines) > 0:
            self.lines[-1].set_data(
                np.append(self.pts[:, 0].flatten(), self.pts[0,0]),
                np.append(self.pts[:, 1].flatten(), self.pts[0,1])
            )

        if len(self.scatters) > 0:
            self.scatters[-1].set_offsets(self.pts)

        if self.state != 'follow':
            self.c.draw()
        else:
            self.c.draw_idle()            

    def followmouse(self, event):
        if self.state in ['follow']:
            p2 = np.array([event.xdata, event.ydata])
            self.update_square(p2)
            self.update_lines()     

    def releaseonclick(self, event):
        if (np.linalg.norm(self.pts[2, :] - self.pts[0, :]) < 20):        
            print('Too close to release.')

        elif (self.pts[2, 0] <= self.rightmost_pt[0, 0] or
              self.pts[2, 1] <= self.rightmost_pt[0, 1]):
            print("Can't release here. Release down / right of : {0}".format(self.rightmost_pt))

        else:            
            #self.rightmost_pt = np.copy(self.pts[2, :]).reshape((1,2))
            self.update_rightmost_point(self.pts[2, :].reshape((1,2)))            
            
            self.update_lines()            
            
            self.square_list.append(self.pts[:,:])

            if self.n_squares() < 3:
                self.update_state(new_state = "add_squares")                
            else:
                self.update_state(new_state = "done")
                self.grab_squares()    
                
    def grab_squares(self):
        # u  - like the x-axis of a square
        # v  - like the y-axis of a square
        # s  - the side-length of a square
        # mu - the center point of a square    
        us = np.zeros((3,2))
        vs = np.zeros((3,2))
        ss = np.zeros((3,1))
        mus = np.zeros((3,2))

        # compute the quantities described above
        for i in range(3):            
            us[i, :] = self.square_list[i][1, :] - self.square_list[i][0, :]            
            vs[i, :] = self.square_list[i][1, :] - self.square_list[i][2, :]
            mus[i, :] = self.square_list[i].mean(axis = 0)
            ss[i, 0] = np.linalg.norm(us[i, :])

        # get the average of each quantity
        u_mean = us.mean(axis = 0)
        v_mean = vs.mean(axis = 0)
        s_mean = ss.mean(axis = 0)
        mu_mean = mus.mean(axis = 0)

        # d  - the average distance between the three squares
        # a  - the average vertical / horizontal displacements between the three squares
        d = 0.5 * ( np.linalg.norm(mus[1, :] - mus[0, :]) +  np.linalg.norm(mus[2, :] - mus[1, :]))
        a = np.sqrt(1 / 2) * d

        # lattice - a 3x3 regular grid of points set up so that the three squares'
        #           centers fall near the diagonal of the lattice
        lattice = np.zeros((9, 2))
        for i in range(3):
            for j in range(3):
                lattice[3 * i + j, :] = (
                    mu_mean + (a/s_mean) * ((-1 + j) *  u_mean + (1 - i) * v_mean)
                )

        # extract the cube faces by grabbing squares centered on the 9 lattice points
        faces = np.zeros((150, 150, 3), dtype=self.img.dtype)
        for i in range(3):
            for j in range(3):
                # Map squares to a reference square
                pts1 = np.array([
                    lattice[3 * i + j, :] + 0.45 * (-u_mean + v_mean),
                    lattice[3 * i + j, :] + 0.45 * ( u_mean + v_mean),
                    lattice[3 * i + j, :] + 0.45 * ( u_mean - v_mean),
                    lattice[3 * i + j, :] + 0.45 * (-u_mean - v_mean)                
                ]).astype('float32')
                pts2 = np.float32([[0,0], [50, 0], [50, 50], [0, 50]])

                # Apply transformation 
                M = cv.getPerspectiveTransform(pts1, pts2)
                dst = cv.warpPerspective(self.img, M, (50, 50))
                faces[50*i : 50*(i+1),  50*j : 50*(j+1), :] = dst

        # Show the results
        self.ax.imshow(faces)
        self.ax.set_title('Press q to accept. Ctrl+z to return to editing.')
        self.c.draw()

        # Retain the extracted faces
        self.faces = faces

if __name__ == '__main__':
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = '/home/publius/Documents/RubiksTest/TestImages/Shot10_2.jpg'
    img = plt.imread(img_path)

    manual_square_extractor = ManualSquareExtractor(img)
    plt.show()
