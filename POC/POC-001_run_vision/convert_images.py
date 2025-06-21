import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def crop_and_resize_image(image_path):
    """
    Crop the image to match 512x256 aspect ratio, then resize it to 512x256.
    
    :param image_path: Path to the image file
    :return: Resized image object
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Get original dimensions
        original_width, original_height = img.size
        target_aspect_ratio = 512 / 256
        
        # Calculate the current aspect ratio
        current_aspect_ratio = original_width / original_height
        
        # Crop the image to match the target aspect ratio
        if current_aspect_ratio > target_aspect_ratio:
            # Crop width (too wide)
            new_width = int(original_height * target_aspect_ratio)
            left = (original_width - new_width) // 2
            right = left + new_width
            img = img.crop((left, 0, right, original_height))
        elif current_aspect_ratio < target_aspect_ratio:
            # Crop height (too tall)
            new_height = int(original_width / target_aspect_ratio)
            top = (original_height - new_height) // 2
            bottom = top + new_height
            img = img.crop((0, top, original_width, bottom))
        
        # Resize the image to 512x256
        resized_img = img.resize((512, 256))
        return resized_img
    except Exception as e:
        print(f"Error: {e}")
        return None

def display_image(image):
    """
    Display the given image using matplotlib.
    
    :param image: Image object to display
    """
    if image:
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        plt.show()
    else:
        print("No image to display.")

def convert_to_yuv420(image):
    """
    Convert a Pillow Image to YUV420 format and separate Y, U, V components.
    
    :param image: Pillow Image object
    :return: Tuple of three numpy arrays (Y, U, V)
    """
    try:
        # Convert image to RGB mode if not already
        image = image.convert('RGB')
        
        # Get image data as numpy array
        rgb_array = np.array(image)
        
        # # Extract R, G, B channels
        # R = rgb_array[:, :, 0]
        # G = rgb_array[:, :, 1]
        # B = rgb_array[:, :, 2]
        
        # # Calculate Y, U, V components
        # Y = 0.299 * R + 0.587 * G + 0.114 * B
        # U = -0.14713 * R - 0.28886 * G + 0.436 * B
        # V = 0.615 * R - 0.51499 * G - 0.10001 * B
        
        # # Downsample U and V components to half resolution (YUV420)
        # U = U[::2, ::2]
        # V = V[::2, ::2]
        
        # # Convert to uint8
        # Y = np.clip(Y, 0, 255).astype(np.uint8)
        # U = np.clip(U, 0, 255).astype(np.uint8)
        # V = np.clip(V, 0, 255).astype(np.uint8)
                
        # Extract R, G, B channels
        R = rgb_array[:, :, 0].astype(np.float32)
        G = rgb_array[:, :, 1].astype(np.float32)
        B = rgb_array[:, :, 2].astype(np.float32)
        
        # Calculate Y, U, V components
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = (B - Y) * 0.493
        V = (R - Y) * 0.877
        
        # Downsample U and V components to half resolution (YUV420)
        U = U[::2, ::2]
        V = V[::2, ::2]
        
        # Convert to uint8
        Y = np.clip(Y, 0, 255).astype(np.uint8).flatten()
        U = np.clip(U + 128, 0, 255).astype(np.uint8).flatten()  # Add 128 for proper centering
        V = np.clip(V + 128, 0, 255).astype(np.uint8).flatten()  # Add 128 for proper centering

        return Y, U, V
    except Exception as e:
        print(f"Error converting to YUV420: {e}")
        return None, None, None


def display_grayscale_image(data, width, height):
    """
    Display a grayscale image using matplotlib.
    
    :param data: uint8 array representing pixel brightness
    :param width: Width of the image
    :param height: Height of the image
    """
    try:
        # Reshape the data to the specified dimensions
        image = np.reshape(data, (len(data) // width, width))
        
        # Display the image using matplotlib
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')  # Hide axes
        plt.show()
    except Exception as e:
        print(f"Error displaying grayscale image: {e}")



def yuv420_to_rgb(Y, U, V, width, height):
    """
    Convert YUV420 format to RGB array.
    
    :param Y: uint8 array representing Y component (brightness)
    :param U: uint8 array representing U component (chrominance)
    :param V: uint8 array representing V component (chrominance)
    :param width: Width of the image
    :param height: Height of the image
    :return: RGB array of shape (height, width, 3)
    """
    try:
        # Reshape Y to (height, width)
        Y = np.reshape(Y, (height, width)).astype(np.float32)
        
        # Reshape U and V to (height // 2, width // 2)
        U = np.reshape(U, (height // 2, width // 2)).astype(np.float32)
        V = np.reshape(V, (height // 2, width // 2)).astype(np.float32)
        
        # Upsample U and V to match Y's dimensions
        U = np.repeat(np.repeat(U, 2, axis=0), 2, axis=1)
        V = np.repeat(np.repeat(V, 2, axis=0), 2, axis=1)
        
        # Convert YUV to RGB
        R = Y + 1.402 * (V - 128)
        G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)
        B = Y + 1.772 * (U - 128)
        
        # Clip values to [0, 255] and convert to uint8
        R = np.clip(R, 0, 255).astype(np.uint8)
        G = np.clip(G, 0, 255).astype(np.uint8)
        B = np.clip(B, 0, 255).astype(np.uint8)
        
        # Stack R, G, B channels to form RGB array
        rgb_array = np.stack((R, G, B), axis=-1)
        
        return rgb_array
    except Exception as e:
        print(f"Error converting YUV420 to RGB: {e}")
        return None

def display_rgb_image(rgb_array, width, height):
    """
    Display an RGB image using matplotlib in Jupyter.
    
    :param rgb_array: RGB array of shape (height, width, 3)
    :param width: Width of the image
    :param height: Height of the image
    """
    try:
        # Reshape the RGB array to the specified dimensions
        rgb_array = np.reshape(rgb_array, (height, width, 3))
        
        # Display the image using matplotlib
        plt.imshow(rgb_array)
        plt.axis('off')  # Hide axes
        plt.show()
    except Exception as e:
        print(f"Error displaying RGB image: {e}")


# Example usage
# # Replace 'your_image_path.jpg' with the actual path to your image
# resized_image = crop_and_resize_image('res/1.jpg')
# y,u,v = convert_to_yuv420(resized_image)
# print(len(y.flatten()), len(u.flatten()), len(v.flatten()))


def loadys(Y, out, width, height):
    MODEL_DIM = width * height
    UV_SIZE = MODEL_DIM // 4
    
    for oy in range(0, height, 1):
        for ox in range(0, width, 8):
            ys = oy * width + ox
            
            if oy & 1 == 0:
                outy0 = 0
                outy1 = UV_SIZE * 2
            else:
                outy0 = UV_SIZE
                outy1 = UV_SIZE * 3

            # Store
            for i in range(0, 8, 2):
                out[outy0 + (oy//2) * (width//2) + ox//2 + i//2] = Y[ys + i]
                out[outy1 + (oy//2) * (width//2) + ox//2 + i//2] = Y[ys + i+1]

        
def loaduv(C, out, offset, width, height):
    for i in range(width//2 * height // 2):
        out[offset + i] = C[i]


def loadys_test():
    WIDTH = 16
    HEIGHT = 4

    buffer_size = WIDTH * HEIGHT * 3 // 2

    # Create buffer
    Ytest = np.zeros(WIDTH * HEIGHT, dtype=np.uint8)
    Utest = np.zeros(WIDTH * HEIGHT // 4, dtype=np.uint8)
    Vtest = np.zeros(WIDTH * HEIGHT // 4, dtype=np.uint8)

    # Fill the test buffer
    for i in range(WIDTH * HEIGHT):
        Ytest[i] = i % 256
    for i in range(WIDTH * HEIGHT // 4):
        Utest[i] = 7
        Vtest[i] = 11
    # print(Ytest)
    # print(Utest)
    # print(Vtest)
    
    outtest = np.zeros(buffer_size, dtype=np.uint8)
    loadys(Ytest, outtest, WIDTH, HEIGHT)
    loaduv(Utest, outtest, WIDTH*HEIGHT, WIDTH, HEIGHT)
    loaduv(Vtest, outtest, WIDTH*HEIGHT+WIDTH*HEIGHT//4, WIDTH, HEIGHT)
    
    print(outtest)
    print(np.reshape(outtest, (buffer_size//WIDTH, WIDTH)))



def prepare(image_path):
    WIDTH = 512
    HEIGHT = 256
    resized_image = crop_and_resize_image(image_path)
    Y, U, V = convert_to_yuv420(resized_image)
    buffer_size = WIDTH * HEIGHT * 3 // 2
    out_frame = np.zeros(buffer_size, dtype=np.uint8)
    loadys(Y, out_frame, WIDTH, HEIGHT)
    loaduv(U, out_frame, WIDTH*HEIGHT, WIDTH, HEIGHT)
    loaduv(V, out_frame, WIDTH*HEIGHT+WIDTH*HEIGHT//4, WIDTH, HEIGHT)
    return out_frame




if __name__ == "__main__":
    # resized_image = crop_and_resize_image('res/1.jpg')
    # y,u,v = convert_to_yuv420(resized_image)
    # print(len(y.flatten()), len(u.flatten()), len(v.flatten()))
    
    # loadys_test()

    buffer_frame = prepare(os.path.join(os.path.dirname(__file__), "res/1.jpg"))
    last_frame = prepare(os.path.join(os.path.dirname(__file__), "res/2.jpg"))
    
    input_frames = np.concatenate((buffer_frame, last_frame))
    print(input_frames)
    