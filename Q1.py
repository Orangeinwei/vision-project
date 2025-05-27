import numpy as np
import cv2
import matplotlib.pyplot as plt
from ca_utils import im2single, single2im

def compute_gradient_magnitude(gr_image):

    # Define Sobel kernels for x and y directions
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1], 
                        [0, 0, 0], 
                        [1, 2, 1]], dtype=np.float32)
    
    # Apply Sobel kernels using filter2D for convolution
    Gx = cv2.filter2D(gr_image.astype(np.float32), -1, sobel_x)
    Gy = cv2.filter2D(gr_image.astype(np.float32), -1, sobel_y)
    
    # Compute gradient magnitude using the Pythagorean theorem
    magnitude = np.sqrt(np.square(Gx) + np.square(Gy))
    
    return magnitude

def compute_gradient_direction(gr_image):
    
    # Define Sobel kernels for x and y directions
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1], 
                        [0, 0, 0], 
                        [1, 2, 1]], dtype=np.float32)
    
    # Apply Sobel kernels using filter2D for convolution
    Gx = cv2.filter2D(gr_image.astype(np.float32), -1, sobel_x)
    Gy = cv2.filter2D(gr_image.astype(np.float32), -1, sobel_y)
    
    # Compute gradient direction using arctan2 (returns angles in radians)
    # arctan2 handles the quadrant correctly based on the signs of Gx and Gy
    direction = np.arctan2(Gy, Gx)
    
    return direction

if __name__ == "__main__":
    # Load Image
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Loading greyscale images
    img = cv2.imread('data/coins.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image 'data/coins.jpg'")
        exit(1)
    
    # Calculate gradient magnitude and direction
    magnitude = compute_gradient_magnitude(img)
    direction = compute_gradient_direction(img)
    
    # Save results
    cv2.imwrite('gradient_magnitude.jpg', magnitude)
    cv2.imwrite('gradient_direction.jpg', direction)

    np.save('gradient_magnitude.npy', magnitude)
    np.save('gradient_direction.npy', direction)
    
    # Visualisation results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    # Normalisation of gradient magnitude for better visualisation
    magnitude_normalized = magnitude / magnitude.max() * 255
    plt.imshow(magnitude_normalized, cmap='jet')
    plt.title('Gradient Magnitude')
    plt.axis('off')
    
    plt.subplot(133)
    # Orientation can be shown using the HSV colour map
    direction_normalized = (direction + np.pi) / (2 * np.pi) * 255
    plt.imshow(direction_normalized, cmap='hsv')
    plt.title('Gradient Direction')
    plt.axis('off')

     # Verify the saved files
    print("Verifying saved numpy arrays:")
    mag_loaded = np.load('gradient_magnitude.npy')
    dir_loaded = np.load('gradient_direction.npy')
    print(f"Magnitude array shape: {mag_loaded.shape}")
    print(f"Direction array shape: {dir_loaded.shape}")
    
    plt.tight_layout()
    plt.savefig('gradient_visualization.jpg')
    plt.show()
    
    print("Gradient computation completed successfully.")
    print("Outputs saved as 'gradient_magnitude.jpg' and 'gradient_direction.jpg'")

   
  