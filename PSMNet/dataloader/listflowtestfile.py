import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','tif','TIF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    test_path = filepath
    test_dir = os.listdir(test_path)

    test_left_img = []
    test_right_img = []

    for dd in test_dir:
        if dd == 'left':
            for im in os.listdir(test_path + '/' + dd):
                if is_image_file(test_path + '/' + dd + im):
                    test_left_img.append(test_path + '/' + dd + '/' + im)
        elif dd == 'right':
            for im in os.listdir(test_path + '/' + dd):
                if is_image_file(test_path + '/' + dd + im):
                    test_right_img.append(test_path + '/' + dd + '/' + im)

    return test_left_img, test_right_img