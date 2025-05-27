import numpy as np
import cv2
import matplotlib.pyplot as plt
from ca_utils import im2single, single2im

def compute_colour_histogram(image, num_bins):
 
    # Ensure image is in the correct format and range
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Extract individual color channels
    # Note: OpenCV uses BGR order, so we need to reverse the channels
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if it's a 3-channel image
        b, g, r = cv2.split(image)
    else:
        raise ValueError("Input must be a 3-channel color image")
    
    # Define histogram range and bin size for 8-bit images
    hist_range = (0, 256)  # Range is inclusive at lower end, exclusive at upper end
    bin_size = 256 // num_bins
    
    # Calculate histograms for each channel
    hist_r = np.zeros(num_bins, dtype=np.float32)
    hist_g = np.zeros(num_bins, dtype=np.float32)
    hist_b = np.zeros(num_bins, dtype=np.float32)
    
    # Iterate through each bin and count pixels
    for i in range(num_bins):
        lower_bound = i * bin_size
        upper_bound = (i + 1) * bin_size if i < num_bins - 1 else 256
        
        hist_r[i] = np.sum((r >= lower_bound) & (r < upper_bound))
        hist_g[i] = np.sum((g >= lower_bound) & (g < upper_bound))
        hist_b[i] = np.sum((b >= lower_bound) & (b < upper_bound))
    
    return hist_r, hist_g, hist_b

if __name__ == "__main__":
    print("Starting main function...")
  
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load color image
    image = cv2.imread('data/flower.jpg')
    if image is None:
        print("Error: Could not load image 'data/flower.jpg'")
        exit(1)
    
    # The image read by OpenCV is in BGR format, which needs to be converted to RGB format for processing and displaying
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate the colour histogram, using 32 bins
    num_bins = 32
    hist_r, hist_g, hist_b = compute_colour_histogram(image_rgb, num_bins)

    hist_r = hist_r.astype(np.int64)
    hist_g = hist_g.astype(np.int64)
    hist_b = hist_b.astype(np.int64)
    
    # Saving Histogram Data
    np.save('histogram_r.npy', hist_r)
    np.save('histogram_g.npy', hist_g)
    np.save('histogram_b.npy', hist_b)
    
    # Visualisation of images and histograms
    plt.figure(figsize=(15, 8))
    
    # Displaying the original image
    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Creating bin centre points for plotting histograms
    bin_edges = np.linspace(0, 256, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Draw Red Channel Histogram
    plt.subplot(2, 2, 2)
    plt.bar(bin_centers, hist_r, width=bin_width*0.8, color='r', alpha=0.7)
    plt.title('Red Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim(0, 255)
    
    # Draw Green Channel Histogramraw 
    plt.subplot(2, 2, 3)
    plt.bar(bin_centers, hist_g, width=bin_width*0.8, color='g', alpha=0.7)
    plt.title('Green Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim(0, 255)
    
    # Draw Blue Channel Histogram
    plt.subplot(2, 2, 4)
    plt.bar(bin_centers, hist_b, width=bin_width*0.8, color='b', alpha=0.7)
    plt.title('Blue Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim(0, 255)

    print("\nHistogram Array Values (first 10 bins):")
    print(f"Red channel histogram (shape: {hist_r.shape}):")
    print(hist_r[:10])
    print(f"Green channel histogram (shape: {hist_g.shape}):")
    print(hist_g[:10])
    print(f"Blue channel histogram (shape: {hist_b.shape}):")
    print(hist_b[:10])
    
    plt.tight_layout()
    plt.savefig('color_histograms.jpg')
    plt.show()
    
    print("Color histogram computation completed successfully.")
    print("Histograms saved as 'histogram_r.npy', 'histogram_g.npy', and 'histogram_b.npy'")