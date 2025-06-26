from scipy.ndimage import rotate as scipy_rotate, shift as scipy_shift, zoom as scipy_zoom
import numpy as np
class AugmentImage:
    def __init__(self, rotate=None, shift=None, zoom=None):
        self.rotate_angle = rotate
        self.shift_val = shift
        self.zoom_factor = zoom

    def rotate(self, img):
        angle = self.rotate_angle
        if angle == "random":
            angle = np.random.uniform(-15, 15)
        return scipy_rotate(img, angle=angle, order=1, mode='nearest')

    def shift(self, img):
        x = y = self.shift_val
        if x == "random":
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)

        if img.ndim == 2:
            shift_vals = (x, y)
        elif img.ndim == 3:
            # Example: (1, 28, 28) → must be (0, x, y)
            shift_vals = (0, x, y)
        else:
            raise ValueError(f"[shift] Unsupported img shape: {img.shape}")

        return scipy_shift(img, shift=shift_vals, order=1, mode='nearest')

    @staticmethod
    @staticmethod
    def zoom_to_same_shape(img, zoom_factor):
        original_shape = img.shape
        zoomed = scipy_zoom(img, zoom=zoom_factor, order=1, mode='nearest')
        if zoomed.ndim == 3:
            _, zh, zw = zoomed.shape
        elif zoomed.ndim == 2:
            zh, zw = zoomed.shape
        else:
            raise ValueError(f"Unexpected image shape: {zoomed.shape}")
        if len(original_shape) == 3:
            _, oh, ow = original_shape
        elif len(original_shape) == 2:
            oh, ow = original_shape
            oc = 1
        if zh > oh:
            start = (zh - oh) // 2
            zoomed = zoomed[start:start + oh, :]
        if zw > ow:
            start = (zw - ow) // 2
            zoomed = zoomed[:, start:start + ow]
        if zoomed.ndim == 3:
            _, zh, zw = zoomed.shape
        elif zoomed.ndim == 2:
            zh, zw = zoomed.shape
        else:
            raise ValueError(f"Unexpected image shape: {zoomed.shape}")
        pad_h = max(oh - zh, 0)
        pad_w = max(ow - zw, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if zoomed.ndim == 2:  # H x W
            padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        elif zoomed.ndim == 3:  # C x H x W
            padding = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        elif zoomed.ndim == 4:  # N x C x H x W (batch mode)
            padding = ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        else:
            raise ValueError(f"Unsupported zoomed shape: {zoomed.shape}")
        zoomed = np.pad(zoomed, padding, mode='constant')
        zoomed = zoomed[:oh, :ow]
        if zoomed.shape != original_shape:
            zoomed = np.resize(zoomed, original_shape)
        return zoomed
    def zoom(self, img):
        factor = self.zoom_factor
        if factor == "random":
            factor = np.random.uniform(0.9, 1.1)
        return self.zoom_to_same_shape(img, factor)
    def augment(self, img):
        assert isinstance(img, np.ndarray), "Image must be a NumPy array"
        original_shape = img.shape
        if self.rotate_angle is not None:
            img = self.rotate(img)
        if self.shift_val is not None:
            img = self.shift(img)
        if self.zoom_factor is not None:
            img = self.zoom(img)
        if img.shape != original_shape:
            print(f"[WARN] Shape mismatch corrected: {img.shape} -> {original_shape}")
            img = np.resize(img, original_shape)
        return img
    def augment_library(self, img_list):
        augmented = []
        for i, img in enumerate(img_list):
            aug = self.augment(img)
            if aug.shape != img.shape:
                print(f"[ERROR] Item {i} has shape {aug.shape}, resizing to {img.shape}")
                aug = np.resize(aug, img.shape)
            augmented.append(aug)
        return np.stack(augmented, axis=0)
def augment_dataset(X, augmenter):
    original_shape = X.shape[1:]
    X_aug = augmenter.augment_library(X)
    return X_aug.reshape(X.shape[0], -1)