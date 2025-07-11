import random
from scipy.ndimage import rotate
import numpy as np
from torchvision import transforms
from PIL import Image

def rotate1(image, label):
    theta = random.uniform(-10, 10)
    new_img = rotate(image, theta, reshape=False, mode='nearest')
    new_lab = rotate(label, theta, reshape=False, order=0, mode='nearest') 
    return new_img, new_lab

def flip1(image, label):
    if random.random() > 0.5: 
        image = np.fliplr(image) 
        label = np.fliplr(label)  
    return image.copy(), label.copy()  

def flip2(image, label):
    if random.random() > 0.5: 
        image = np.flipud(image)  
        label = np.flipud(label)  
    return image.copy(), label.copy()  

def augment1(image, label):
    """
    Randomly applies a combination of transformations to the image and label.
    :param image: Image (e.g., medical image)
    :param label: Corresponding label (e.g., segmentation mask)
    :return: Transformed image and label
    """
    if len(image.shape) == 2:  # Ensure it's 2D
        available_transformations = {'flip': flip1, 'flipv': flip2, 'rotate': rotate1}
        num_transformations_to_apply = random.randint(0, len(available_transformations))
        
        transformed_image = image
        transformed_label = label
        
        if num_transformations_to_apply == 0:
            return image, label  # Return original if no transformations
        
        # Apply the selected transformations
        for _ in range(num_transformations_to_apply):
            key = random.choice(list(available_transformations))
            transformed_image, transformed_label = available_transformations[key](transformed_image, transformed_label)
        
        return transformed_image, transformed_label  # Return transformed image and label
    else:
        raise Exception('Invalid dimensions for image augmentation - only supported for 2D images')
