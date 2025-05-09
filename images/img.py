import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image path
image_path = "/content/bill-1.png"

# Check file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

# Open and convert to RGB
with Image.open(image_path) as image:
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image_np = np.array(image)  # Convert to NumPy array for OpenCV

# Convert to grayscale
gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

# OPTION 1: Simple thresholding (better for sharp text with good contrast)
# Adjust the threshold value (127) to control darkness - higher value = more dark pixels
_, binary_simple = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

# OPTION 2: Improved adaptive thresholding with smaller block size
# Smaller block size (11) and higher C value (8) for more precise text edges
binary_adaptive = cv2.adaptiveThreshold(
    gray, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 
    blockSize=11,  # Smaller block size for finer detail
    C=8  # Adjusted C value
)

# OPTION 3: Otsu's thresholding (good for bimodal images like text documents)
# This automatically finds the optimal threshold value
_, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Optional: Apply light sharpening to enhance text edges
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened = cv2.filter2D(binary_adaptive, -1, kernel)

# Save the processed image
output_path = "image_enhanced.jpg"
cv2.imwrite(output_path, binary_adaptive)  # Choose which version to save
print(f"Processed image saved to {output_path}")

# Show original and processed images
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 2)
plt.title("Original image")
plt.imshow(image_np)
plt.axis("off")


plt.subplot(2, 2, 3)
plt.title("Adaptive Threshold (Improved)")
plt.imshow(binary_adaptive, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Original Grayscale")
plt.imshow(gray, cmap='gray')
plt.axis("off")


plt.tight_layout()
plt.show()


#-------------------------------------------------------------------------------

import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image path
image_path = "/content/bill-1.png"

# Check file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

# Open and convert to RGB
with Image.open(image_path) as image:
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image_np = np.array(image)  # Convert to NumPy array for OpenCV

# Convert to grayscale
gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

# Create a balanced adaptive threshold with medium text thickness
# Medium blockSize and lower C value for more substantial text
balanced_adaptive = cv2.adaptiveThreshold(
    gray, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 
    blockSize=13,  # Medium block size (between original 15 and improved 11)
    C=4  # Lower C value for thicker text while maintaining clarity
)

# Optional: Apply a very mild dilation to make text slightly thicker without bloating
kernel = np.ones((2, 2), np.uint8)
text_enhanced = cv2.dilate(balanced_adaptive, kernel, iterations=1)

# Create variations for comparison
# Version 1: Even more balanced (alternative parameters)
balanced_alt = cv2.adaptiveThreshold(
    gray, 255, 
    cv2.ADAPTIVE_THRESH_MEAN_C,  # Using MEAN instead of GAUSSIAN can sometimes give clearer text
    cv2.THRESH_BINARY, 
    blockSize=15,
    C=5
)


# Display original and balanced version
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image_np)  
plt.axis("off")

plt.tight_layout()
plt.show()

# Compare different balancing approaches
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Grayscale")
plt.imshow(gray, cmap='gray')
plt.axis("off")

output_path = "gray.jpg"
cv2.imwrite(output_path, gray)  # Choose which version to save
print(f"Processed image saved to {output_path}")


plt.subplot(1, 3, 2)
plt.title("Gaussian Adaptive (C=4)")
plt.imshow(balanced_adaptive, cmap='gray')
plt.axis("off")

# Save the processed image
output_path = "image_balanced.jpg"
cv2.imwrite(output_path, balanced_adaptive)  # Choose which version to save
print(f"Processed image saved to {output_path}")


plt.tight_layout()
plt.show()

#--------------------------------------------------

import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image path
image_path = "/content/bill-1.png"

# Check file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

# Open and convert to RGB
with Image.open(image_path) as image:
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image_np = np.array(image)  # Convert to NumPy array for OpenCV

# Create a copy of the image for processing
result = image_np.copy()

# Custom thresholding in RGB space
# Define the thresholds
dark_threshold = 50
light_threshold = 230

# Apply custom thresholding to each pixel
# For each pixel, if any channel is below dark_threshold, set to 0
# If all channels are above light_threshold, set to 255
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        pixel = result[i, j]
        
        # Check if any channel is dark (below dark_threshold)
        if np.any(pixel <= dark_threshold):
            result[i, j] = [0, 0, 0]  # Set to black
        
        # Check if all channels are light (above light_threshold)
        elif np.all(pixel >= light_threshold):
            result[i, j] = [255, 255, 255]  # Set to white

# Vectorized version (much faster)
# Create a new array for better performance
result_fast = image_np.copy()

# Create masks
dark_mask = np.any(image_np <= dark_threshold, axis=2)
light_mask = np.all(image_np >= light_threshold, axis=2)

# Apply transformations
result_fast[dark_mask] = [0, 0, 0]
result_fast[light_mask] = [255, 255, 255]

# Save the processed image
output_path = "custom_threshold_image.jpg"
cv2.imwrite(output_path, cv2.cvtColor(result_fast, cv2.COLOR_RGB2BGR))  # OpenCV uses BGR
print(f"Processed image saved to {output_path}")

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image_np)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title(f"Custom Threshold (Dark ≤{dark_threshold}, Light ≥{light_threshold})")
plt.imshow(result_fast)
plt.axis("off")

# Create a grayscale version of the result for comparison
gray_result = cv2.cvtColor(result_fast, cv2.COLOR_RGB2GRAY)
plt.subplot(1, 3, 3)
plt.title("Grayscale Result")
plt.imshow(gray_result, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

