import numpy as np
import cv2 as cv
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.distance import cdist
from skmpe import mpe


def contour_extraction(input_image_path):
    im = cv.imread(input_image_path)
    assert im is not None, "file could not be read, check with os.path.exists()"
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    # cv.drawContours(im, contours, -1, (0, 255, 0), 3)
    # cv.imshow('Contours', im)

    return contours[0]


def uniformly_sample_contour(contour_array, Ndesiredpoints, visualise=0):
    # Ensure the contour is closed (duplicate the first point at the end)
    if not np.array_equal(contour_array[0], contour_array[-1]):
        contour_array = np.vstack([contour_array, contour_array[0]])

    # Calculate the arc length of the contour
    arc_length = cv.arcLength(contour_array, closed=True)

    # Initialize the list to store accumulated distances (arc length at each point)
    accumulated_lengths = [0]

    # Compute the accumulated contour length at each point
    for i in range(1, len(contour_array)):
        segment_length = np.linalg.norm(contour_array[i] - contour_array[i - 1])
        accumulated_lengths.append(accumulated_lengths[-1] + segment_length)

    # Normalize accumulated lengths to get equally spaced query points
    query_lengths = np.linspace(0, arc_length, Ndesiredpoints, endpoint=False)

    # Interpolate the x and y coordinates separately using the accumulated lengths
    x_interp = interp1d(
        accumulated_lengths,
        contour_array[:, 0],
        kind="linear",
        fill_value="extrapolate",
    )
    y_interp = interp1d(
        accumulated_lengths,
        contour_array[:, 1],
        kind="linear",
        fill_value="extrapolate",
    )

    # Get the uniformly sampled points
    sampled_x = x_interp(query_lengths)
    sampled_y = y_interp(query_lengths)

    # Combine x and y coordinates into an array
    sampled_points = np.column_stack((sampled_x, sampled_y))

    # Visualize the sampled points if requested
    if visualise == 1:
        plt.figure(figsize=(6, 6))
        plt.scatter(sampled_points[:, 0], sampled_points[:, 1], color="red", s=1)
        plt.axis("equal")
        plt.show()

    return sampled_points, arc_length


def compute_laplacian_coordinates(contour: np.ndarray, scales) -> np.ndarray:
    """
    Compute Laplacian coordinates for a 2D contour at multiple scales.

    Parameters:
        contour (np.ndarray): Nx2 array of 2D contour points.
        step (int): Integer step size to define scales.

    Returns:
        np.ndarray: SxNx2 array of Laplacian coordinates for each scale and contour point.
    """

    S = len(scales)
    N = len(contour)

    # Output array to store Laplacian coordinates (S scales, N points, 2D coordinates)
    laplacian_coords = np.zeros((S, N, 2))

    # Precompute indices for adjacency matrix
    left_indices = np.arange(N)  # Left neighbors
    right_indices = (np.arange(N) + 1) % N  # Right neighbors (cyclic)

    for s_idx, scale in enumerate(scales):
        # Create adjacency matrix using broadcasting (for all neighbors in one step)
        adjacency = np.zeros((N, N))
        for offset in range(1, scale + 1):
            adjacency[
                left_indices, (left_indices - offset) % N
            ] = 1  # Neighbors on the left (cyclic)
            adjacency[
                right_indices, (right_indices + offset) % N
            ] = 1  # Neighbors on the right (cyclic)

        # Compute the degree matrix
        degree = np.sum(adjacency, axis=1)
        degree_matrix_inv = np.diag(1 / degree)  # Inverse of the degree matrix

        # Compute the Laplacian operator
        laplacian = (
            np.eye(N) - degree_matrix_inv @ adjacency
        )  # Efficient matrix multiplication

        # Scale the Laplacian operator
        laplacian /= 2 * scale + 1

        # Compute Laplacian coordinates by multiplying contour points with the Laplacian
        laplacian_coords[s_idx] = laplacian @ contour  # Efficient matrix multiplication

    return laplacian_coords


def plot_laplacian_vectors_for_scale(
    contour_points: np.ndarray, laplacian_vectors: np.ndarray, scale_idx: int
):
    """
    Plot the 2D contour points with Laplacian vectors for a specific scale.

    Parameters:
        contour_points (np.ndarray): Nx2 array of 2D contour points.
        laplacian_vectors (np.ndarray): SxNx2 array of Laplacian vectors for each scale and point.
        scale_idx (int): Index of the scale to visualize.
    """
    if scale_idx < 0 or scale_idx >= laplacian_vectors.shape[0]:
        raise ValueError("Invalid scale index. Must be within the range of scales.")

    plt.figure(figsize=(8, 8))
    plt.scatter(
        contour_points[:, 0], contour_points[:, 1], color="blue", label="Contour Points"
    )

    for i in range(contour_points.shape[0]):
        x, y = contour_points[i]
        dx, dy = laplacian_vectors[scale_idx, i] * 10
        # print(laplacian_vectors)
        plt.arrow(
            x, y, dx, dy, color="red", head_width=0.02, head_length=0.03, alpha=0.6
        )

    plt.title(f"Laplacian Vectors for Scale Index {scale_idx}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.legend()
    plt.show()


def project_laplacians_on_tangent_normal(laplacian_vectors, contour):
    # Number of scales (S) and contour points (N)
    S, N, _ = laplacian_vectors.shape

    # Initialize the surfaces
    localXlaplaciansurface = np.zeros((S, N))
    localYlaplaciansurface = np.zeros((S, N))

    # Compute tangent and normal vectors for the contour
    tangents, normals = compute_tangent_normal(contour, plot=0)

    # Project each laplacian vector onto the tangent and normal
    for s in range(S):
        for n in range(N):
            laplacian_vector = laplacian_vectors[s, n]

            # Project Laplacian vector onto the tangent (x direction)
            localXlaplaciansurface[s, n] = np.dot(laplacian_vector, tangents[n])

            # Project Laplacian vector onto the normal (y direction)
            localYlaplaciansurface[s, n] = np.dot(laplacian_vector, normals[n])

    return localXlaplaciansurface, localYlaplaciansurface


def compute_tangent_normal(contour, plot=False):
    """
    Calculates tangent and normal vectors for each point in a contour, considering 4 neighbors.

    Args:
      contour: A NumPy array representing the contour (Nx2).
      plot (optional): Boolean flag to indicate plotting (default: False).

    Returns:
      tangents: A NumPy array of shape (Nx2) containing tangent vectors.
      normals: A NumPy array of shape (Nx2) containing normal vectors.
    """

    # Number of points in the contour
    N = contour.shape[0]

    # Initialize arrays for tangent and normal vectors
    tangents = np.zeros_like(contour)
    normals = np.zeros_like(contour)

    # Loop over all contour points (wrap around at the ends for a closed contour)
    for i in range(N):
        # Get indices of neighboring points (considering wrapping)
        prev2_index = (i - 2) % N
        prev_index = (i - 1) % N
        next_index = (i + 1) % N
        next2_index = (i + 2) % N

        # Get neighboring points
        prev2_point = contour[prev2_index]
        prev_point = contour[prev_index]
        next_point = contour[next_index]
        next2_point = contour[next2_index]

        # Calculate average tangent vector
        tangent = (
            prev2_point - next2_point
        ) / 2.0  # Average of two neighbors on each side

        # Normalize the tangent vector
        tangent /= np.linalg.norm(tangent)

        # Calculate normal vector (perpendicular to tangent, outward direction)
        normal = np.array([-tangent[1], tangent[0]])  # Perpendicular to tangent

        # Assign the tangent and normal vectors
        tangents[i] = tangent
        normals[i] = normal

    if plot:
        # Plot the contour, tangents, and normals
        plt.figure(figsize=(6, 6))
        plt.plot(contour[:, 0], contour[:, 1], "b-", label="Contour")  # Contour in blue

        visualization_scale = 100

        # Plot tangent vectors in red
        for i in range(N):
            plt.quiver(
                contour[i, 0],
                contour[i, 1],
                tangents[i, 0] * visualization_scale,
                tangents[i, 1] * visualization_scale,
                color="red",
                scale=10,
                angles="xy",
                scale_units="xy",
                width=0.005,
            )

        # Plot normal vectors in green
        for i in range(N):
            plt.quiver(
                contour[i, 0],
                contour[i, 1],
                normals[i, 0] * visualization_scale,
                normals[i, 1] * visualization_scale,
                color="green",
                scale=10,
                angles="xy",
                scale_units="xy",
                width=0.005,
            )

        plt.gca().set_aspect("equal", adjustable="box")
        plt.legend()
        plt.title("Contour with Tangent and Normal Vectors")
        plt.show()

    return tangents, normals


def visualize_laplacian_surfaces(localXlaplaciansurface, localYlaplaciansurface):
    S, N = localXlaplaciansurface.shape  # Get the shape of the surfaces

    # Create an array representing contour points (0 to N-1)
    contour_points = np.arange(N)

    # Create an array representing scales (0 to S-1)
    scales = np.arange(S)

    # Create a meshgrid for the plot
    X, Y = np.meshgrid(contour_points, scales)

    # Plotting the Local X Laplacian Surface (projection on tangent)
    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X, Y, localXlaplaciansurface, cmap="viridis")
    ax1.set_title("Local X Laplacian Surface (Tangent Projections)")
    ax1.set_xlabel("Contour Point (n)")
    ax1.set_ylabel("Scale (s)")
    ax1.set_zlabel("Laplacian Value (Local X)")

    # Plotting the Local Y Laplacian Surface (projection on normal)
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(X, Y, localYlaplaciansurface, cmap="inferno")
    ax2.set_title("Local Y Laplacian Surface (Normal Projections)")
    ax2.set_xlabel("Contour Point (n)")
    ax2.set_ylabel("Scale (s)")
    ax2.set_zlabel("Laplacian Value (Local Y)")

    plt.tight_layout()
    plt.show()


def compute_Fsurface(laplacian_surfacec, laplacian_surfacet, lambda_val, scales):
    """
    Compute F surface for Fast Marching Method using OpenCV.

    Parameters:
    - laplacian_surfacec: numpy array of shape (S, Nc)  # Control surface
    - laplacian_surfacet: numpy array of shape (S, Nt)  # Target surface
    - lambda_val: regularization parameter (scalar)
    - scales: numpy array of length S (scales for each dimension S)

    Returns:
    - Fsurface: numpy array of shape (Nc, Nt), the computed F surface
    """
    S, Nc = laplacian_surfacec.shape
    _, Nt = laplacian_surfacet.shape

    n_times = 2
    error_stack = np.zeros((S, Nc, Nt))

    for k in range(S):
        scale = scales[k]

        # Compute pairwise distances efficiently
        error = (
            cdist(
                laplacian_surfacec[k, :].reshape(-1, 1),
                laplacian_surfacet[k, :].reshape(-1, 1),
            )
            / scale
        )

        kernel_size = int(n_times * scale) + 1
        x, y = np.mgrid[-scale : scale + 1, -scale : scale + 1]
        gaussian = np.exp(-(x**2 + y**2) / (2 * scale**2))

        # Define kernel with Gaussian weighting
        B = np.zeros((kernel_size, kernel_size))
        B[:scale, :scale] = 1
        B[scale + 1 :, scale + 1 :] = 1
        kernel = B * gaussian

        # if k>5:
        #     plot_surface(kernel)

        # Efficient convolution using OpenCV
        error_extended = np.tile(error, (3, 3))
        kconv_error = cv.filter2D(error_extended, -1, kernel)
        error_stack[k, :, :] = kconv_error[Nc : 2 * Nc, Nt : 2 * Nt]

    # Combine all scales and compute the F surface
    aux_fsurface = 1.0 / (np.sum(error_stack, axis=0) + lambda_val)
    Fsurface = aux_fsurface

    # plot_surface(Fsurface)

    return Fsurface


def compute_shortest_path(F, lambda_val, visualise=0):
    """
    Computes the shortest path and visualizes it using the mpe method for a given speed surface.

    Parameters:
    - F: 2D numpy array representing the speed surface.
    """

    # Apply decay to the F-surface
    F = np.tile(F, (3, 3))

    # Create a meshgrid of indices
    x, y = np.meshgrid(np.arange(F.shape[1]), np.arange(F.shape[0]))

    # Calculate distances from the origin (0, 0)
    distances = np.sqrt(x**2 + y**2)

    # Normalize distances (optional)
    max_distance = np.sqrt(F.shape[1] ** 2 + F.shape[0] ** 2)
    normalized_distances = distances / max_distance

    # Apply linear decay (adjust lambda_val as needed)
    decay_factor = 1 - (lambda_val * normalized_distances)
    decay_factor = decay_factor * np.max(F) * 2
    F = F + decay_factor

    # Automatically define start and end points based on the size of F
    start_point = (0, 0)  # Starting point (top-left corner)
    end_point = (F.shape[0] - 1, F.shape[1] - 1)  # Ending point (bottom-right corner)

    # Compute the shortest path using mpe
    path_info = mpe(F, start_point, end_point)

    # Extract computed travel time for the given ending point and extracted path
    travel_time = path_info.pieces[0].travel_time
    path = path_info.path

    if visualise:
        # Get the shape of the speed surface
        nrows, ncols = F.shape

        # Create a meshgrid for visualization
        xx, yy = np.meshgrid(np.arange(ncols), np.arange(nrows))

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(F, cmap="gray", alpha=0.9)  # Visualize the speed surface (F)
        ax.plot(
            path[:, 1],
            path[:, 0],
            "-",
            color=[0, 1, 0],
            linewidth=2,
            label="Shortest Path",
        )
        ax.plot(start_point[1], start_point[0], "or", label="Start Point")
        ax.plot(end_point[1], end_point[0], "o", color=[1, 1, 0], label="End Point")

        # Plot travel time contours
        tt_c = ax.contour(xx, yy, travel_time, 20, cmap="plasma", linewidths=1.5)
        ax.clabel(tt_c, inline=1, fontsize=9, fmt="%d")

        # Set title and axis off
        ax.set_title("Travel Time Contours and Minimal Path")
        ax.axis("off")

        # Colorbar for the travel time
        cb = fig.colorbar(tt_c)
        cb.ax.set_ylabel("Travel Time")

        # Show the plot
        plt.legend()
        plt.show()

    # print(path)
    # actual_path = path[:, [1, 0]]
    return path


def plot_surface(F, plot_name="Surface Plot"):
    """
    Plots a 3D surface using matplotlib.

    Parameters:
    - F: 2D numpy array, the surface to plot.
    - plot_name: string, the title of the plot.
    """
    # Generate grid for plotting
    x = np.arange(F.shape[1])
    y = np.arange(F.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, F, cmap="viridis", edgecolor="none")

    # Add title, labels, and color bar
    ax.set_box_aspect((1, 1, 1))  # Equal aspect ratio for x, y, z axes
    ax.view_init(elev=90, azim=0)

    ax.set_title(plot_name)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Show the plot
    plt.show()


def interpolate_contour_point(contour, index):
    """
    Interpolates a point on a contour.

    Args:
      contour: A 2D NumPy array representing the contour (Nx2).
      index: A 1D NumPy array of indices to interpolate.

    Returns:
      A 2D NumPy array representing the interpolated points (Mx2),
      where M is the number of indices.
    """

    # Handle out-of-bounds indices (clamp to valid range)
    index = np.clip(index, 0, len(contour) - 1)

    # Extract integer and fractional parts of the index
    int_index = np.floor(index).astype(int)
    frac_index = index - int_index

    # Handle edge cases (first and last points)
    next_index = np.minimum(int_index + 1, len(contour) - 1)

    # Interpolate x and y coordinates
    interpolated_x = contour[int_index, 0] + frac_index * (
        contour[next_index, 0] - contour[int_index, 0]
    )
    interpolated_y = contour[int_index, 1] + frac_index * (
        contour[next_index, 1] - contour[int_index, 1]
    )

    return np.column_stack((interpolated_x, interpolated_y))


def plot_interpolated_contours(new_contour_c, new_contour_t):
    """
    Plots interpolated contours with corresponding points.

    Args:
      new_contour_c: Interpolated contour C as a NumPy array (Nx2).
      new_contour_t: Interpolated contour T as a NumPy array (Nx2).

    Returns:
      None (displays the plot directly).
    """

    plt.plot(new_contour_c[:, 0], new_contour_c[:, 1], "bo-", label="new_contour_c")
    plt.plot(new_contour_t[:, 0], new_contour_t[:, 1], "ro-", label="new_contour_t")

    for i in range(len(new_contour_c)):
        plt.plot(
            [new_contour_c[i, 0], new_contour_t[i, 0]],
            [new_contour_c[i, 1], new_contour_t[i, 1]],
            "gray",
            linewidth=0.5,
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Interpolated Contours")
    plt.legend()
    plt.axis("equal")
    plt.show()


def plot_interpolated_contours_color(new_contour_c, new_contour_t, plot_title):
    """
    Plots interpolated contours with corresponding points, sharing a color code.

    Args:
      new_contour_c: Interpolated contour C as a NumPy array (Nx2).
      new_contour_t: Interpolated contour T as a NumPy array (Nx2).

    Returns:
      None (displays the plot directly).
    """
    num_points = len(new_contour_c)
    colors = cm.viridis(
        np.linspace(0, 1, num_points)
    )  # Generate colors from a colormap

    for i in range(num_points - 1):
        # Plot segments for contour C
        plt.plot(
            new_contour_c[i : i + 2, 0],
            new_contour_c[i : i + 2, 1],
            color=colors[i],
            linewidth=2,
        )
        # Plot segments for contour T
        plt.plot(
            new_contour_t[i : i + 2, 0],
            new_contour_t[i : i + 2, 1],
            color=colors[i],
            linewidth=2,
        )
        # Plot lines connecting corresponding points
        plt.plot(
            [new_contour_c[i, 0], new_contour_t[i, 0]],
            [new_contour_c[i, 1], new_contour_t[i, 1]],
            color=colors[i],
            linewidth=0.5,
            linestyle="--",
        )

    # Plot the last connection (if necessary)
    plt.plot(
        [new_contour_c[-1, 0], new_contour_t[-1, 0]],
        [new_contour_c[-1, 1], new_contour_t[-1, 1]],
        color=colors[-1],
        linewidth=0.5,
        linestyle="--",
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(plot_title)
    plt.axis("equal")
    plt.show()


def procrustes_transform(new_contour_c, new_contour_t):
    """
    Computes and applies a rigid Procrustes transformation to align new_contour_t to new_contour_c.

    Args:
        new_contour_c (np.ndarray): Target contour (Nx2).
        new_contour_t (np.ndarray): Contour to transform (Nx2).

    Returns:
        np.ndarray: Transformed version of new_contour_t aligned with new_contour_c.
    """
    # Center the contours
    c_mean = np.mean(new_contour_c, axis=0)
    t_mean = np.mean(new_contour_t, axis=0)
    centered_c = new_contour_c - c_mean
    centered_t = new_contour_t - t_mean

    # Singular Value Decomposition (SVD) for optimal rotation
    U, _, Vt = np.linalg.svd(np.dot(centered_t.T, centered_c))
    R = np.dot(U, Vt)  # Rotation matrix

    # Apply rotation and translation to align contours
    transformed_t = np.dot(centered_t, R)

    return transformed_t, centered_c


if __name__ == "__main__":
    # input_image_path = "shapes/square.bmp"  # Replace with your input image
    # target_image_path = "shapes/rectangle.bmp"  # Replace with your target image

    input_image_path = "shapes/60_06.bmp"  # Replace with your input image
    target_image_path = "shapes/60_13.bmp"  # Replace with your target image

    current_contour = contour_extraction(input_image_path)
    target_contour = contour_extraction(target_image_path)

    Ndesiredpointscurr = 200

    current_contour_array = current_contour.reshape(-1, 2).astype(np.float32)
    current_contour_array[:, 1] = -current_contour_array[
        :, 1
    ]  # cv img coord sign conventions...
    sampled_contour_c, current_length = uniformly_sample_contour(
        current_contour_array, Ndesiredpointscurr, visualise=1
    )

    # simulate badly conditioned extarcted contour indexes
    offset = 40
    sampled_contour_c = np.roll(sampled_contour_c, offset, axis=0)

    target_contour_array = target_contour.reshape(-1, 2).astype(np.float32)
    target_contour_array[:, 1] = -target_contour_array[:, 1]
    target_length = cv.arcLength(target_contour_array, closed=True)
    Ndesiredpointstarg = round(Ndesiredpointscurr * target_length / current_length)
    sampled_contour_t, target_length = uniformly_sample_contour(
        target_contour_array, Ndesiredpointstarg, visualise=1
    )

    # init contours for comparison purposes v.s. homogeneous method
    sampled_contour_c_init = sampled_contour_c
    sampled_contour_t_init, _ = uniformly_sample_contour(
        target_contour_array, Ndesiredpointscurr, visualise=0
    )

    N = sampled_contour_c.shape[0]
    Nt = sampled_contour_t.shape[0]
    # number of scales
    Ns = 5
    s_max = min(N, Nt) // 5
    s_min = 1
    scales = np.linspace(s_min, s_max, Ns, dtype=int)
    print("Scale vec (indexes):", scales)

    laplacian_vectors_c = compute_laplacian_coordinates(sampled_contour_c, scales)
    (
        localXlaplaciansurface_c,
        localYlaplaciansurface_c,
    ) = project_laplacians_on_tangent_normal(laplacian_vectors_c, sampled_contour_c)
    laplacian_vectors_t = compute_laplacian_coordinates(sampled_contour_t, scales)
    (
        localXlaplaciansurface_t,
        localYlaplaciansurface_t,
    ) = project_laplacians_on_tangent_normal(laplacian_vectors_t, sampled_contour_t)

    # visualize_laplacian_surfaces(localXlaplaciansurface_c, localYlaplaciansurface_c)
    # visualize_laplacian_surfaces(localXlaplaciansurface_t, localYlaplaciansurface_t)

    lambda_val = 0.5
    FsurfaceX = compute_Fsurface(
        localXlaplaciansurface_c, localXlaplaciansurface_t, lambda_val, scales
    )
    FsurfaceY = compute_Fsurface(
        localYlaplaciansurface_c, localYlaplaciansurface_t, lambda_val, scales
    )

    # plot_surface(FsurfaceY, "FsurfaceY")
    F = FsurfaceX**2 + FsurfaceY**2
    F = F / np.max(F)
    map = compute_shortest_path(F, lambda_val, visualise=1)

    N, M = F.shape
    closest_index = np.argmin(np.abs(map[:, 1] - M + 1))
    closest_index_2 = np.argmin(np.abs(map[:, 1] - M * 2 + 1))

    retrieved_offset = int(map[closest_index, 0] % N)
    print("Our method's retrieved contour offset:", retrieved_offset)

    sampled_contour_c = np.roll(sampled_contour_c, -retrieved_offset, axis=0)

    map[:, 0] = map[:, 0] - map[closest_index, 0]
    map[:, 1] = map[:, 1] - M
    map = map[closest_index : closest_index_2 + 1, :]

    new_contour_c = interpolate_contour_point(sampled_contour_c, map[:, 0])
    new_contour_t = interpolate_contour_point(sampled_contour_t, map[:, 1])

    aligned_t, new_contour_c = procrustes_transform(new_contour_c, new_contour_t)
    aligned_t[:, 0] = aligned_t[:, 0] + np.max(new_contour_c[:, 0]) * 2.5

    plot_interpolated_contours_color(new_contour_c, aligned_t, "elastic mapping (ours)")

    aligned_t_init, sampled_contour_c_init = procrustes_transform(
        sampled_contour_c_init, sampled_contour_t_init
    )
    aligned_t_init[:, 0] = (
        aligned_t_init[:, 0] + np.max(sampled_contour_c_init[:, 0]) * 2.5
    )
    plot_interpolated_contours_color(
        sampled_contour_c_init, aligned_t_init, "homogeneous mapping"
    )

