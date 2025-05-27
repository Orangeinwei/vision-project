import numpy as np
import matplotlib.pyplot as plt

def compute_transform_matrix(points, theta, scale, translation):
   
    # Extract translation parameters
    tx, ty = translation
    
    # Convert rotation angle from degrees to radians
    theta_rad = np.deg2rad(theta)
    
    # Calculate the center of the points
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    
    # Create translation matrix to move points to origin
    T_to_origin = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ])
    
    # Create rotation matrix (counter-clockwise rotation)
    R = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0],
        [np.sin(theta_rad), np.cos(theta_rad), 0],
        [0, 0, 1]
    ])
    
    # Create scaling matrix
    S = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ])
    
    # Create translation matrix to move points back from origin
    T_from_origin = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])
    
    # Create final translation matrix
    T_final = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    
    # Combine all transformations: First translate to origin, then rotate, scale, translate back, and apply final translation
    transform_matrix = T_final @ T_from_origin @ S @ R @ T_to_origin
    
    return transform_matrix

if __name__ == "__main__":
    # Load points from numpy file
    points = np.load('data/points.npy')
    
    # Define transformation parameters
    theta = 45  # 45 degrees rotation
    scale = 1.5  # Scale by 1.5
    translation = (20, 30)  # Translate by (20, 30)
    
    # Compute transformation matrix
    transform_matrix = compute_transform_matrix(points, theta, scale, translation)
    
    # Save the matrix to a file
    np.save('transform_matrix.npy', transform_matrix)
    print("Transformation matrix saved as 'transform_matrix.npy'")
    
    # Visualize original and transformed points
    # Convert points to homogeneous coordinates
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # Apply transformation
    transformed_points_homogeneous = homogeneous_points @ transform_matrix.T
    
    # Convert back to Cartesian coordinates
    transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2:]
    
    # For visualization, only plot the first 500 points
    sample_size = 500
    
    # Print the transformation matrix
    print("Transformation Matrix:")
    print(transform_matrix)

    plt.figure(figsize=(10, 8))
    plt.scatter(points[:sample_size, 0], points[:sample_size, 1], color='blue', label='Original Points', alpha=0.6)
    plt.scatter(transformed_points[:sample_size, 0], transformed_points[:sample_size, 1], color='red', label='Transformed Points', alpha=0.6)
    plt.title('Original vs Transformed Points')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('transformation_visualization.jpg')
    plt.show()
    
