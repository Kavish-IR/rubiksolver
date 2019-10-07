import sys
import numpy as np
from extract_squares_from_image import extract_cube_faces_from_stream, capture_faces, load_imgs_from_dir, capture_faces_from_images
from color_classifier import get_classifier, label_images
from Cube import Cube
from Solver import Solver
from SolutionGallery import SolutionGallery
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle
from matplotlib.image import AxesImage
from matplotlib import get_backend
import cv2 as cv
from manual_square_extractor import ManualSquareExtractor


class ClickableRectangle:
    def __init__(self, fig, ax, colors_cube, f, i_f, j_f):
        self.f   = f
        self.i_f = i_f
        self.j_f = j_f

        self.colors_cube = colors_cube
        self.x = i_f
        self.y = j_f
        self.color = self.colors_cube[f, i_f, j_f]

        self.fig = fig
        self.ax = ax

        self.rect = Rectangle(
                (self.x, self.y),
                width = 1,
                height = 1,
                facecolor = self.color,
                linewidth=2,
                edgecolor='k',
                picker=True
        )
        
        self.patch = self.ax.add_patch(self.rect)

        self.active = False        
        clicker = self.fig.canvas.mpl_connect('button_press_event', lambda e: self.onclick(e))
        presser = self.fig.canvas.mpl_connect('key_press_event',    lambda e: self.keypress(e))

    def onclick(self, event):
        # Was the click on the same axis as the rectangle?
        if event.inaxes != self.rect.axes:
            self.active = False
            return
            
        # Was the click inside the rectangle?
        contains, attrd = self.rect.contains(event)
        if not contains:
            self.active = False
            return

        # Only concerned with double click events
        if event.dblclick:
            # Set active
            self.active = True

    def keypress(self, event):
        if not self.active:
            return

        elif event.key in ['w','W']:
            self.color = 'white'
        elif event.key in ['o', 'O']:
            self.color = 'orange'
        elif event.key in ['r', 'R']:
            self.color = 'red'
        elif event.key in ['g', 'G']:
            self.color = 'green'
        elif event.key in ['b', 'B']:
            self.color = 'blue'
        elif event.key in ['y', 'Y']:
            self.color = 'yellow'
        elif event.key == 'enter':
            self.active = False

        self.colors_cube[self.f, self.i_f, self.j_f] = self.color
        
        self.patch.set_facecolor(self.color)
        self.fig.canvas.draw()                                           



class RectContainer:
    def __init__(self, fig, ax_img, ax_squares, f, orig_img, faces, colors_cube, clf_yuv):
        self.fig = fig
        self.ax_img = ax_img
        self.ax_squares = ax_squares
        self.f = f        
        self.orig_img = orig_img
        self.faces = faces
        self.colors_cube = colors_cube
        self.clf_yuv = clf_yuv

        # Plot the extracted faces on the left of the image        
        self.ax_img.set_xlim(0, 150)
        self.ax_img.set_ylim(0, 150)
        self.ax_img.axis('equal')
        self.ax_img.axis('off')
        self.ax_img.imshow(self.faces)
        img_clicker = self.fig.canvas.mpl_connect('button_press_event', lambda e: self.onclick(e))        
                
        # Plot the colors on the right of the image
        self.clickable_rects = []        
        for s in range(9):
            i_f = s % 3
            j_f = int(s / 3)
            cr = ClickableRectangle(self.fig, self.ax_squares, self.colors_cube, self.f, i_f, j_f)
            self.clickable_rects.append(cr)
            
        self.ax_squares.set_xlim(0, 3)
        self.ax_squares.set_ylim(0, 3)
        self.ax_squares.axis('equal')
        self.ax_squares.axis('off')
        self.fig.canvas.draw()
        

    def update_squares(self):
        # Plot the colors on the right of the image
        for s in range(9):
            i_f = s % 3
            j_f = int(s / 3)
            cr = self.clickable_rects[s]
            cr.patch.set_facecolor(self.colors_cube[self.f, i_f, j_f])
        
    def onclick(self, event):
        if event.inaxes != self.ax_img.axes:
            return
        
        if event.dblclick:

            mse = ManualSquareExtractor(self.orig_img)
            plt.show()

            while mse.complete is False:
                plt.draw()                
                plt.pause(0.5)

            self.faces = mse.faces

            pred_colors_yuv, pred_proba_yuv = label_images(
                self.clf_yuv,
                [self.faces.reshape(1, 150, 150, 3)],
                faces_per_image = 1
            )
            pred_colors_yuv2 = np.array(pred_colors_yuv).reshape((1,3,3))

            self.colors_cube[self.f, :, :] = pred_colors_yuv2
            
            del mse

            self.ax_img.imshow(self.faces)
            self.update_squares()
            self.fig.canvas.draw()

            
def onpick(event):
    if isinstance(event.artist, Rectangle):
        patch = event.artist
        print('onpick patch:', patch.get_path())
        patch.set_edgecolor('lime')
        event.canvas.draw()

    elif isinstance(event.artist, AxesImage):                  
        im = event.artist
        A = im.get_array()
        print('onpick image', A.shape)


def check_images(captured_imgs, captured_faces, colors_cube, clf_yuv):
    fig, axs = plt.subplots(2,6, figsize = (24,6))

    for f in range(6):

        RectContainer(
            fig,
            axs[0][f],
            axs[1][f],
            f,
            cv.cvtColor(captured_imgs[f], cv.COLOR_BGR2RGB),
            captured_faces[f],
            colors_cube,
            clf_yuv
        )

    fig.tight_layout()

    return 


def main(input_dir = None):

    # Input  the faces of the Cube
    if input_dir is None:        
        captured_faces, captured_imgs = capture_faces()            
    else:
        input_imgs = load_imgs_from_dir(input_dir)
        captured_faces, captured_imgs = capture_faces_from_images(input_imgs)
    

    # Get the color classifier
    clf_yuv = get_classifier()

    # Predict the face colors from the input images
    faces = np.array(captured_faces)
    pred_colors_yuv, pred_proba_yuv = label_images(clf_yuv, [faces])
    colors_cube = np.array(pred_colors_yuv).reshape((6,3,3))

    # Inspect / adjust results if necessary. This step can modify pred_colors_yuv2.
    check_images(captured_imgs, captured_faces, colors_cube, clf_yuv)
    plt.show()

    # Define the cube using the updated colors
    c = Cube(colors_cube)
    
    # Solve and retain moves
    initial_state = c.export_state()
    s = Solver(c)
    s.solve()
    solve_moves = c.recorded_moves
    
    # Display the solution
    sg = SolutionGallery(initial_state, solve_moves)
    plt.show()
    
    

if __name__ == '__main__':
    if len(sys.argv) > 1:
        img_dir = sys.argv[1]
        main(img_dir)
    else:
        main()
