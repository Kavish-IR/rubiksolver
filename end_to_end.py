from extract_squares_from_image import extract_cube_faces_from_stream, capture_faces
from color_classifier import get_classifier, label_images
import numpy as np
from Cube import Cube
from Solver import Solver
from SolutionGallery import SolutionGallery
import matplotlib.pyplot as plt

#def main():
#captured_faces, captured_imgs = extract_cube_faces_from_stream()
captured_faces, captured_imgs = capture_faces()
clf_yuv = get_classifier()

# Flips
captured_faces[4] = np.swapaxes(captured_faces[4], 1, 0)
captured_faces[4] = np.flip(captured_faces[4], 0)


captured_faces[5] = np.swapaxes(captured_faces[5], 1, 0)
captured_faces[5] = np.flip(captured_faces[5], 1)


faces = np.array(captured_faces)
pred_colors_yuv, pred_proba_yuv = label_images(clf_yuv, [faces])

pred_colors_yuv2 = np.array(pred_colors_yuv).reshape((6,3,3))

plt.close('all')

c = Cube(pred_colors_yuv2)
c.cube_plot()
c.square_plot()

fig, axs = plt.subplots(2, 3)
for i in range(2):
    for j in range(3):
        k = 3 * i + j
        axs[i][j].imshow(np.flipud(captured_faces[k]))

plt.show()        

# Solve and retain moves
initial_state = c.export_state()
s = Solver(c)
s.solve()
solve_moves = c.recorded_moves

# Display the solution
sg = SolutionGallery(initial_state, solve_moves)
plt.show()

#print(initial_state)
    
    
