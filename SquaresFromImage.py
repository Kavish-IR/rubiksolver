import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy import spatial
import numpy.random as rng
import time
import sys

def capture_image(cap = None):
    if cap is None:
        cap = cv.VideoCapture(0)

    # Read from the camera feed and create window
    fps = cap.get(cv.CAP_PROP_FPS)
    ret, frame = cap.read()
    cv.imshow('Input Stream', frame)

    # Bring camera feed to front
    cv.setWindowProperty('Input Stream', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.setWindowProperty('Input Stream', cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
    
    while(True):
        ret, frame = cap.read()

        # Show the frame capture
        cv.imshow('Input Stream', frame)
        
        # Handle Key Presses
        key_press = cv.waitKey(1)
        
        # Take a picture if space is pressed
        if key_press % 256 == 32:
            exit_program = 0
            break
        
        # Quit the program if q is pressed
        if key_press & 0xFF == ord('q'):
            exit_program = 1
            break
        
    return frame, exit_program


def hipass_filter_img(img):
    ksize = 21
    blur = cv.GaussianBlur(img, (ksize,ksize), 0)
    img_highpass = img - blur
    img_highpass = cv.medianBlur(img_highpass, 5)
    
    return img_highpass


def filter_regions(img_thresh):
    ret, markers = cv.connectedComponents(img_thresh, connectivity=4)
    max_marker = np.max(markers.flatten())
    marker_counts, bins = np.histogram(markers.flatten(), bins = np.arange(max_marker+1))
    
    for i in range(len(marker_counts)):
        if i < len(marker_counts) - 1:
            if marker_counts[i] < 100:
                markers[markers == i] = 0
        else:
            markers[markers >= i] = 0
            
    marker_counts,_ = np.histogram(markers.flatten(), bins = np.arange(max_marker+1))
    j = 0
    for i in range(len(marker_counts)):
        if marker_counts[i] > 0:
            markers[markers == i] = j
            j += 1

    return markers


def contiguous_regions(img_thresh):
    ret, markers = cv.connectedComponents(255 - img_thresh, connectivity=8)
    max_marker = np.max(markers.flatten())
    marker_counts, bins = np.histogram(markers.flatten(), bins = np.arange(max_marker+1))
    
    for i in range(len(marker_counts)):
        if i < len(marker_counts) - 1:
            if marker_counts[i] < 10 * 10:
                markers[markers == i] = 0
            elif marker_counts[i] > 80 * 80:
                markers[markers == i] = 0
        else:
            markers[markers >= i] = 0
            
    marker_counts,_ = np.histogram(markers.flatten(), bins = np.arange(max_marker+1))
    j = 0
    for i in range(len(marker_counts)):
        if marker_counts[i] > 0:
            markers[markers == i] = j
            j += 1

    return markers

def fill_internal_holes(marked_region):
    # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    marked_region_ff = marked_region.copy()
    h,w = marked_region_ff.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(marked_region_ff, mask, (0,0), 255)
    marked_region_ff_inv = cv.bitwise_not(marked_region_ff)
    marked_region_out = marked_region | marked_region_ff_inv
    
    return marked_region_out
   
def markers_to_quad_j(markers, j):
    marked_region = np.where(markers == j, 255, 0).astype('uint8')
    marked_region2 = fill_internal_holes(marked_region)

    # Get contours
    cnts, _ = cv.findContours(marked_region2.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts    = sorted(cnts, key=cv.contourArea, reverse=True)

    # Convex hulls
    hulls = [cv.convexHull(c, False) for c in cnts]

    # loop over our contours and approximate hull by a polynomial
    # if the polynomail is a quadrilateral that isn't too big, we will
    # keep the result for downstream processing
    quads = []
    areas = []
    peris = []
    for c in hulls:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, epsilon = 0.1 * peri, closed = True)
        deg = len(approx)
        
        if deg == 4 and peri < 4 * 85:
            area = cv.contourArea(approx)
            peri = cv.arcLength(approx, True)            
            peris.append(peri)
            areas.append(area)
            quads.append(approx)
           
    # get the quadrilateral with the largest area
    if quads:
        quad = quads[areas.index(max(areas))]
    else:
        quad = []
        
    return quad, marked_region2

def markers_to_quads(markers):
    max_marker = np.max(markers)
    quads = []
    marked_regions2 = []
    
    for j in range(max_marker):
        quad, m = markers_to_quad_j(markers, j)
        
        if len(quad) > 0:
            quads.append(quad.squeeze())        
            marked_regions2.append(m)

    mm = np.zeros_like(markers)
    for m in marked_regions2:
        mm += m

    return quads, mm

def is_likely_square(quad):
    ps = np.squeeze(quad)
    vs = np.zeros((4,2))
    ls = np.zeros((4,1))

    for i in range(4):
        vs[i, :] = ps[(i+1)%4, :] - ps[i,:]
        ls[i] = np.linalg.norm(vs[i,:])

    ratio = np.std(ls) / np.mean(ls)
    
    return ratio < .10

def reorder_square(square):
    mean = np.mean(square, axis=0)
    delta = square - mean
    angles = np.arctan2(delta[:,1], delta[:,0]).reshape(4,1)
    angles = (360 / (2*np.pi)) * np.mod(angles,2*np.pi)
    
    square = np.append(square, angles, axis=1)
    square = square[square[:,2].argsort()]
    square = square[:,0:2].astype('int32')

    return square


def filter_by_area(squares):
    n_sq = len(squares)
    
    # Re-order squares by area
    squares = sorted(squares, key=cv.contourArea, reverse=False)    
    square_areas   = np.array([cv.contourArea(sq) for sq in squares])
    area_excl_mean = np.zeros(n_sq)

    # List of indices for squares which we exclude because their
    # area is too large / too small relative to the median of the
    # other remaining squares
    excl_list = []
    
    for i in range(n_sq):
        idxs = [k for k in range(n_sq) if k not in excl_list and k != i]
        area_median_excl_i  = np.median(square_areas[idxs])
        ratio = square_areas[i] /  area_median_excl_i
        
        if ratio > 1.3 or ratio < 0.7:
            excl_list.append(i)
        
    return [squares[i] for i in range(n_sq) if i not in excl_list]


def square_orientation(square):
    horiz = 0.5 * ( (square[0,:] - square[1,:]) + (square[3,:] - square[2,:]) )
    vert  = 0.5 * ( (square[3,:] - square[0,:]) + (square[2,:] - square[1,:]) )
    axes = np.zeros((2,2))
    axes[:,0] = horiz
    axes[:,1] = vert
    return axes
    

def aligned_axes(squares):
    n_sq = len(squares)#square_axes.shape[0]

    # Deal with alignment 
    square_axes  = np.array([square_orientation(sq) for sq in squares])
    square_axes = square_axes.reshape(n_sq, 4)

    # Average
    ax_avg = np.mean(square_axes, axis=0)
    x_ax_avg = ax_avg[0:2]
    y_ax_avg = ax_avg[2:4]

    return x_ax_avg, y_ax_avg


def recoordinatize(squares, quads):
    n_squares = len(squares)
    
    # Compute the square means, and average them to get an origin
    square_means = np.array([np.mean(ps, axis=0).squeeze() for ps in squares])    
    origin = np.mean(square_means, axis=0)

    # Displace the square means
    displaced_square_means = square_means - origin

    # Get the average un-normalized axis alignment
    x_ax_avg, y_ax_avg = aligned_axes(squares)
    x_ax_norm = np.linalg.norm(x_ax_avg)
    y_ax_norm = np.linalg.norm(y_ax_avg)
    avg_ax_norm   = 0.5 * (x_ax_norm + y_ax_norm)

    # Get orthogonal directions
    u_dir = x_ax_avg / x_ax_norm
    v_dir = np.array([-u_dir[1], u_dir[0]])
    basis = np.array( [u_dir, v_dir ]).T

    # Coordinatize the displaced means
    uv_means = np.dot(displaced_square_means, basis)
    u_means = sorted(uv_means[:,0])
    v_means = sorted(uv_means[:,1])

    # Identify jumps in u coordinates - clusters
    u_jump_idx = [0]
    for i in range(1, n_squares):
        jump = (u_means[i] - u_means[i-1]) / avg_ax_norm
        if jump > .6:
            u_jump_idx.append(i)                        
    u_jump_idx.append(n_squares)

    # Get the means within each u cluster
    u_cluster_means = [ np.mean( u_means[u_jump_idx[i-1] : u_jump_idx[i]]) for i in range(1, len(u_jump_idx))]
    u_cluster_means = np.array(u_cluster_means)

    # Identify jumps in v coordinates - clusters
    v_jump_idx = [0]
    for i in range(1, n_squares):
        jump = (v_means[i] - v_means[i-1]) / avg_ax_norm
        if jump > .6:
            v_jump_idx.append(i)                        
    v_jump_idx.append(n_squares)

    # Get the means within each v cluster
    v_cluster_means = [ np.mean( v_means[v_jump_idx[i-1] : v_jump_idx[i]]) for i in range(1, len(v_jump_idx))]
    v_cluster_means = np.array(v_cluster_means)

    # Scale the cluster means by the face length
    u_cluster_means_scaled = u_cluster_means / (1.3 * x_ax_norm)
    v_cluster_means_scaled = v_cluster_means / (1.3 * y_ax_norm)

    # Scaled cluster mean diffs
    u_cluster_means_scaled_diff = np.diff(u_cluster_means_scaled)
    v_cluster_means_scaled_diff = np.diff(v_cluster_means_scaled)

    if len(u_cluster_means) < 3 or len(v_cluster_means) < 3:
        # Compute the square means
        quad_means = np.array([np.mean(ps, axis=0).squeeze() for ps in quads])    
        displaced_quad_means = quad_means - origin
        uv_means_quads = np.dot(displaced_quad_means, basis)
        u_means_quads  = uv_means_quads[:,0]
        v_means_quads  = uv_means_quads[:,1]

        if len(u_cluster_means) == 2:
            # if the means are far apart, we're missing the middle column
            if u_cluster_means_scaled_diff[0] > 1.5:
                u_cluster_means = list(u_cluster_means)
                u_cluster_means.insert(1, 0.5 * (u_cluster_means[0] + u_cluster_means[1]))
                u_cluster_means = np.array(u_cluster_means)

            # if the means are close, we're missing either left or right
            else:
                right_candidate_pos = np.max(u_cluster_means) + (1.3 * x_ax_norm)
                left_candidate_pos  = np.min(u_cluster_means) - (1.3 * x_ax_norm)

                scaled_dist_to_right_cand = [abs(right_candidate_pos - uu) / x_ax_norm for uu in u_means_quads]
                scaled_dist_to_left_cand  = [abs(left_candidate_pos - uu)  / x_ax_norm for uu in u_means_quads]

                right_min_scaled_dist = min(scaled_dist_to_right_cand)
                left_min_scaled_dist  = min(scaled_dist_to_left_cand)                
                
                if min(scaled_dist_to_right_cand) < .1 and right_min_scaled_dist < left_min_scaled_dist:
                    right_idx = scaled_dist_to_right_cand.index(right_min_scaled_dist)
                    u_candid = u_means_quads[right_idx]
                    u_cluster_means = list(u_cluster_means)
                    u_cluster_means.append(u_candid)
                    u_cluster_means = np.array(u_cluster_means)
                    
                elif min(scaled_dist_to_left_cand) < .1:
                    left_idx = scaled_dist_to_left_cand.index(left_min_scaled_dist)
                    u_candid = u_means_quads[left_idx]
                    u_cluster_means = list(u_cluster_means)                                        
                    u_cluster_means.insert(0, u_candid)
                    u_cluster_means = np.array(u_cluster_means)

        if len(v_cluster_means) == 2:
            # if the means are far apart, we're missing the middle column
            if v_cluster_means_scaled_diff[0] > 1.5:
                v_cluster_means = list(v_cluster_means)
                v_cluster_means.insert(1, 0.5 * (v_cluster_means[0] + v_cluster_means[1]))
                v_cluster_means = np.array(v_cluster_means)

            # if the means are close, we're missing either top or bottom
            else:
                up_candidate_pos   = np.max(v_cluster_means) + (1.3 * y_ax_norm)
                down_candidate_pos = np.min(v_cluster_means) - (1.3 * y_ax_norm)

                scaled_dist_to_up_cand   = [abs(up_candidate_pos - vv) / y_ax_norm for vv in v_means_quads]
                scaled_dist_to_down_cand = [abs(down_candidate_pos  - vv) / y_ax_norm for vv in v_means_quads]

                up_min_scaled_dist = min(scaled_dist_to_up_cand)
                down_min_scaled_dist  = min(scaled_dist_to_down_cand)                
                
                if min(scaled_dist_to_up_cand) < .1 and up_min_scaled_dist < down_min_scaled_dist:
                    up_idx = scaled_dist_to_up_cand.index(up_min_scaled_dist)
                    v_candid = v_means_quads[up_idx]
                    v_cluster_means = list(v_cluster_means)
                    v_cluster_means.append(v_candid)
                    v_cluster_means = np.array(v_cluster_means)
                    
                elif min(scaled_dist_to_down_cand) < .1:
                    down_idx = scaled_dist_to_down_cand.index(down_min_scaled_dist)
                    v_candid = v_means_quads[down_idx]
                    v_cluster_means = list(v_cluster_means)                                        
                    v_cluster_means.insert(0, v_candid)
                    v_cluster_means = np.array(v_cluster_means)

    # grid of cluster means
    uv_cluster_means = np.meshgrid(u_cluster_means, v_cluster_means)
    uv_cluster_means = np.array(list(zip(uv_cluster_means[0].flatten(), uv_cluster_means[1].flatten())))

    # Recoodinatize the cluster means back to original coordinates
    cluster_means_out = np.dot(uv_cluster_means, basis.T) + origin

    # Generate squares about the recoodinatized cluster means
    squares_out = []
    for uv_cntr in uv_cluster_means:
        s = np.zeros((4,2))
        s[0,:] = uv_cntr + 0.45 * avg_ax_norm * np.array([-1, 1])
        s[1,:] = uv_cntr + 0.45 * avg_ax_norm * np.array([ 1 ,1])
        s[2,:] = uv_cntr + 0.45 * avg_ax_norm * np.array([ 1,-1])
        s[3,:] = uv_cntr + 0.45 * avg_ax_norm * np.array([-1,-1])
        s = np.dot(s, basis.T) + origin
        squares_out.append(s)

    return cluster_means_out, squares_out


def extract_faces(img, squares):
    rows,cols,ch = img.shape
    faces = np.zeros((3 * 50, 3 * 50, ch), dtype='uint8')
    
    for i in range(3):
        for j in range(3):
            imgcp = np.copy(img)

            # Map squares to a reference square
            pts1 = squares[3 * i + j].astype('float32')
            pts2 = np.float32([[0,50], [50,50], [50, 0], [0,0]])
            M = cv.getPerspectiveTransform(pts1, pts2)
            dst = cv.warpPerspective(imgcp, M, (cols, rows))

            # Retain the result
            faces[50*i:50*(i+1), 50*j:50*(j+1)] = dst[0:50, 0:50].astype('uint8')
            
    return np.flipud(faces)


def plot_contours(contours, ax=None, color='g'):
    if ax == None:
        fig, ax = plt.subplots()

    for c in contours:
        n_pts = len(c)
        c = np.squeeze(c)
        for i in range(0,n_pts):
            p_0 = c[i,:].flatten()
            p_1 = c[(i+1)%n_pts,:].flatten()
            ax.plot(np.array([p_0[0], p_1[0]]), np.array([p_0[1], p_1[1]]), c=color)

            
def process_frame(img_bgr):
    # Convert to RGB and rescale
    img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    nx, ny = img.shape[1], img.shape[0]
    nx_in, ny_in = img.shape[1], img.shape[0]
    nx, ny = 256, int(ny_in * (256.0 / nx_in))
    img = cv.resize(img, (nx,ny))

    # High-pass filter the image
    img_filtered = hipass_filter_img(img)
    img_filtered_gray = cv.cvtColor(img_filtered, cv.COLOR_RGB2GRAY)
    
    # Threshold the high-pass filtered image
    ret, img_thresh = cv.threshold(img_filtered_gray, 127, 255, cv.THRESH_BINARY)

    # Close image
    kernel = np.ones((3,3), np.uint8)
    img_close = cv.morphologyEx(img_thresh, cv.MORPH_CLOSE, kernel)
    
    # Filter regions by size to remove some noise from thresholded img
    #markers = filter_regions(img_thresh)
    markers = filter_regions(img_close)
    img_thresh2 = np.where(markers > 0, 255, 0)
    img_thresh2 = img_thresh2.astype(np.uint8)

    # Dilate image
    kernel = np.ones((3,3), np.uint8)
    img_dilate = cv.dilate(img_thresh2, kernel, iterations=1)

    # Split dilated image into contiguous regions
    markers = contiguous_regions(img_dilate)

    # Filter contiguous regions into regions whos convex hulls are well approximated by quadrilaterals
    quads, mm = markers_to_quads(markers)
    quad_means = np.array([np.mean(ps, axis=0).squeeze() for ps in quads])

    # Filter quadrilaterals to approximately square regions
    squares = list(filter(is_likely_square, quads))
    squares = [reorder_square(s) for s in squares]
    square_means = np.array([np.mean(ps, axis=0).squeeze() for ps in squares])
    
    # Filter squares by area
    filtered_squares = filter_by_area(squares)
    filtered_square_means = np.array([np.mean(ps, axis=0).squeeze() for ps in filtered_squares])
    
    # Orientation?
    x_ax_avg, y_ax_avg = aligned_axes(filtered_squares)

    # Recoordinatize
    recoord_square_means, recoord_squares = recoordinatize(filtered_squares, quads)

    # Extract faces
    faces = extract_faces(img, recoord_squares)

    # All demo display data
    display_data = {'img' : img, 'img_filtered' : img_filtered, 'img_filtered_gray' : img_filtered_gray,
                    'img_thresh' : img_thresh,  'img_close' : img_close, 'img_thresh2' : img_thresh2,
                    'img_dilate' : img_dilate,  'markers' : markers, 'quads' : quads, 'quad_means' : quad_means,
                    'squares' : squares, 'square_means' : square_means, 'filtered_squares' : filtered_squares,
                    'filtered_square_means' : filtered_square_means, 'recoord_squares' : recoord_squares,
                    'recoord_square_means' : recoord_square_means, 'faces' : faces}
    
    return faces, display_data
    
    
def display_result(display_data, axs):
    # Turn off ticks on all plots
    for ax_row in axs:
        for ax in ax_row:
            ax.clear()
            ax.set_xticks([],[])
            ax.set_yticks([],[])    

    n_plot = len(axs)
    m_plot = len(axs[0])
    
    # Plot original
    i_plot = 0
    ax=axs[ int(i_plot / m_plot) ][ i_plot % m_plot]
    ax.imshow(display_data['img'])
    ax.set_title('img')
    
    # Plot filtered image
    i_plot += 1
    ax=axs[ int(i_plot / m_plot) ][ i_plot % m_plot ]   
    ax.imshow(255 - display_data['img_filtered'])
    ax.set_title('img_filtered')

    # Plot grayscale version of filtered image
    i_plot += 1
    ax=axs[ int(i_plot / m_plot) ][ i_plot % m_plot ]       
    ax.imshow(display_data['img_filtered_gray'], cmap='Greys')
    ax.set_title('img_filtered_gray')
    
    # Plot thresholded grayscale image
    i_plot += 1
    ax=axs[ int(i_plot / m_plot) ][ i_plot % m_plot ]       
    ax.imshow(display_data['img_thresh'], cmap='Greys')
    ax.set_title('img_thresh')

    # Plot morphological closure of grayscale image
    i_plot += 1
    ax=axs[ int(i_plot / m_plot) ][ i_plot % m_plot ]           
    ax.imshow(display_data['img_thresh'], cmap='Greys')
    ax.set_title('img_close')

    # Plot threshold of morphological closure image
    i_plot += 1    
    ax=axs[ int(i_plot / m_plot) ][ i_plot % m_plot ]               
    ax.imshow(display_data['img_thresh2'], cmap='Greys')
    ax.set_title('img_thresh2')

    # Dilate the thresholded image to close up holes
    i_plot += 1
    ax=axs[ int(i_plot / m_plot) ][ i_plot % m_plot ] 
    ax.imshow(display_data['img_dilate'], cmap='Greys')
    ax.set_title('img_dilate')

    # Plot the connected components of the last image
    i_plot += 1
    ax=axs[ int(i_plot / m_plot) ][ i_plot % m_plot ] 
    ax.imshow(display_data['markers'], cmap='jet')
    ax.set_title('markers')

    # Plot the boundaries of regions whose convex hulls are
    # approximately quadrilateral
    i_plot += 1
    ax=axs[ int(i_plot / m_plot) ][ i_plot % m_plot ]     
    ax.imshow(np.zeros_like(display_data['img']), cmap='Greys')
    plot_contours(display_data['quads'], ax=ax, color='w')
    ax.scatter(display_data['quad_means'][:,0],
               display_data['quad_means'][:,1], c='r')
    ax.set_title('quads')

    # Plot the boundaries of regions whos convex hulls are
    # approximately squares
    i_plot += 1
    ax=axs[ int(i_plot / m_plot) ][ i_plot % m_plot ]     
    ax.imshow(np.zeros_like(display_data['img']), cmap='Greys')
    plot_contours(display_data['squares'], ax=ax, color='w')
    ax.scatter(display_data['square_means'][:,0],
               display_data['square_means'][:,1], c='r')
    ax.set_title('squares')    

    # Plot squares that are close enough in size
    i_plot += 1
    ax=axs[ int(i_plot / m_plot) ][ i_plot % m_plot ]         
    ax.imshow(np.zeros_like(display_data['img']), cmap='Greys')
    plot_contours(display_data['filtered_squares'], ax=ax, color='w')
    ax.scatter(display_data['filtered_square_means'][:,0],
               display_data['filtered_square_means'][:,1], c='r')
    ax.set_title('filtered_squares')

    # recoordinatized
    i_plot += 1
    ax=axs[ int(i_plot / m_plot) ][ i_plot % m_plot ]
    ax.imshow(np.zeros_like(display_data['img']), cmap='Greys')
    plot_contours(display_data['filtered_squares'], ax=ax, color='w')    
    ax.scatter(display_data['filtered_square_means'][:,0],
               display_data['filtered_square_means'][:,1], c='r')
    plot_contours(display_data['recoord_squares'], ax=ax, color='y')        
    ax.scatter(display_data['recoord_square_means'][:,0],
               display_data['recoord_square_means'][:,1], c='C1')
    ax.set_title('recoordinatized')

    # recoordinatized
    i_plot += 1
    ax=axs[ int(i_plot / m_plot) ][ i_plot % m_plot ]
    ax.imshow(cv.cvtColor(display_data['img'], cv.COLOR_RGB2GRAY), cmap='Greys_r')        
    plot_contours(display_data['recoord_squares'], ax=ax, color='lime')        
    ax.scatter(display_data['recoord_square_means'][:,0],
               display_data['recoord_square_means'][:,1], c='lime')
    ax.set_title('recoordinatized')    

    # extract faces
    i_plot += 1
    ax=axs[ int(i_plot / m_plot) ][ i_plot % m_plot ]
    ax.imshow(display_data['faces'])
    for i in range(3):
        ax.plot([0, 150], [50 * i, 50 * i], c='k')
        ax.plot([50 * i, 50 * i], [0, 150], c='k')        
    ax.set_xlim([0,150])
    ax.set_ylim([0,150])
    ax.set_title('faces')
    
    plt.draw()


#########################################################################
# Plot the result 
n_plot   = 3
m_plot   = 5
fig, axs = plt.subplots(n_plot, m_plot, figsize=(5 * m_plot, 5 * n_plot))


################################################################################
# Read original image
if len(sys.argv) <= 1:
    plt.ion()
    plt.show()
    cap = cv.VideoCapture(0)

    while True:
        img_bgr, exit_program = capture_image(cap)
        
        if exit_program:
            break    
        else:
            faces, display_data = process_frame(img_bgr)
            display_result(display_data, axs)
        
    plt.ioff()    
    
else:
    img_bgr = cv.imread(sys.argv[1], 3)
    process_frame(img_bgr, axs)
    plt.show()

