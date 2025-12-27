from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random


def image_blurring(image, blur_radius):
    try:
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    except:
        return image

def image_rotation(image, angle):
    try:
        return image.rotate(angle)
    except:
        return image

def image_flipping(image, direction):
    try:
        if direction == "horizontal":
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == "vertical":
            return image.transpose(Image.FLIP_TOP_BOTTOM)
    except:
        return image

def image_shifting(image, direction, length):
    try:
        w, h = image.size
        if direction == 'up':
            translation = (0, -length)
        elif direction == 'down':
            translation = (0, length)
        elif direction == 'left':
            translation = (-length, 0)
        elif direction == 'right':
            translation = (length, 0)        
        shifted_image = image.transform(
            (w, h), 
            Image.AFFINE, 
            (1, 0, translation[0], 0, 1, translation[1]),
            fillcolor=(0, 0, 0)
        )
        return shifted_image
    except:
        return image

def image_cropping(image, scale=0.9):
    try:
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        right = left + new_w
        bottom = top + new_h
        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image
    except:
        return image

def image_erasing(image, erase_l=50, erase_w=50):
    try:
        w, h = image.size
        erase_l = min(erase_l, h)
        erase_w = min(erase_w, w)
        top_left_x = random.randint(0, w - erase_w)
        top_left_y = random.randint(0, h - erase_l)
        erased_image = image.copy()
        erase_area = Image.new("RGB", (erase_w, erase_l), (0, 0, 0))
        erased_image.paste(erase_area, (top_left_x, top_left_y))
        return erased_image
    except:
        return image

def adjust_brightness(image, factor):
    try:
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    except:
        return image

def adjust_contrast(image, factor):
    try:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    except:
        return image

def gaussian_noise(image, degree):
    try:
        image_array = np.array(image)
        mean = 0
        std_dev = degree * 255
        noise = np.random.normal(mean, std_dev, image_array.shape)
        noisy_image = image_array + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)
    except:
        return image

def dropout(image, p):
    try:
        image_array = np.array(image)
        mask = np.random.rand(*image_array.shape[:2]) > p
        mask = np.expand_dims(mask, axis=-1)
        dropped_image = image_array * mask
        return Image.fromarray(dropped_image.astype(np.uint8))
    except:
        return image

def salt_and_pepper(image, p):
    try:
        image_array = np.array(image, dtype=np.uint8)
        noise = np.random.rand(image_array.shape[0], image_array.shape[1])
        salt_mask = noise < (p / 2)
        pepper_mask = noise > 1 - (p / 2)
        salt_mask = np.expand_dims(salt_mask, axis=-1).repeat(3, axis=-1)
        pepper_mask = np.expand_dims(pepper_mask, axis=-1).repeat(3, axis=-1)
        image_array[salt_mask] = 255
        image_array[pepper_mask] = 0
        return Image.fromarray(image_array)
    except Exception as e:
        return image

def image_sharpen(image, degree):
    try:
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(degree)
    except:
        return image

def blur_image(image, radius):
    """Alias for image_blurring for backward compatibility"""
    return image_blurring(image, radius)

