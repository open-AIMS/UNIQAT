"""
Comprehensive metrics for underwater image quality assessment.

This module implements various quality metrics specifically designed for
underwater and waterfall images affected by the "blue water problem".
"""

import warnings

import cv2
import numpy as np
from scipy import ndimage

warnings.filterwarnings('ignore')


METRIC_NAMES_37 = (
    'blue_channel_mean', 'green_channel_mean', 'red_channel_mean', 'blue_dominance',
    'green_dominance', 'blue_green_ratio', 'blue_water_severity', 'color_temperature',
    'color_channel_std', 'rms_contrast', 'michelson_contrast', 'local_contrast',
    'weber_contrast', 'lab_contrast', 'sharpness_laplacian', 'sharpness_gradient',
    'blur_estimate', 'edge_density', 'multi_scale_edge_density', 'corner_density',
    'keypoint_density', 'texture_complexity', 'visibility_score', 'turbidity_score',
    'uciqe_score', 'uiqm_score', 'entropy_gray', 'entropy_color',
    'histogram_uniformity_b', 'histogram_uniformity_g', 'histogram_uniformity_r',
    'dynamic_range_b', 'dynamic_range_g', 'dynamic_range_r', 'hue_diversity',
    'feature_usefulness_score', 'marine_science_value',
)

# ---------------------------------------------------------------------------
#  Custom metric registry
# ---------------------------------------------------------------------------
# Lets users add their own metrics without editing this module. A registered
# function receives the UnderwaterMetrics instance, so it can reuse the decoded
# image in any colour space (image, image_rgb, image_gray, image_lab, image_hsv)
# and any built-in metric, and must return a float.
#
#     from uniqat.core.metrics import register_metric, UnderwaterMetrics
#
#     @register_metric('green_red_ratio')
#     def green_red_ratio(m):
#         return float(m.image[:, :, 1].mean() / (m.image[:, :, 2].mean() + 1e-6))
#
#     UnderwaterMetrics(image_path='frame.jpg').calculate_all_metrics()['green_red_ratio']
#
# Custom metrics are appended to calculate_all_metrics() output, after the 37
# built-ins. They are additive only: the built-in names and their order are
# unchanged, so downstream code and the 37-wide deep-learning heads are
# unaffected. The deep models predict the 37 built-ins only; a custom metric is
# computed by the traditional pipeline and has no learned counterpart.

_CUSTOM_METRICS = {}


def register_metric(name, fn=None):
    """Register a custom metric under `name`.

    Usable as a decorator or called directly. Raises ValueError if the name
    collides with a built-in metric, so a typo cannot silently replace one.
    """
    if name in METRIC_NAMES_37:
        raise ValueError(
            f"'{name}' is a built-in metric; choose another name so the "
            f"built-in cannot be silently overwritten"
        )

    def _register(func):
        _CUSTOM_METRICS[name] = func
        return func

    return _register(fn) if fn is not None else _register


def unregister_metric(name):
    """Remove a previously registered custom metric. Returns True if removed."""
    return _CUSTOM_METRICS.pop(name, None) is not None


def registered_metrics():
    """Names of all currently registered custom metrics."""
    return sorted(_CUSTOM_METRICS)

class UnderwaterMetrics:
    """
    A comprehensive collection of underwater image quality metrics.

    Metrics include:
    - Blue water problem detection (color cast analysis)
    - Contrast measurements (global, local, Michelson)
    - Sharpness and blur estimation
    - Feature richness (edge density, corner detection)
    - Visibility and turbidity estimation
    - UCIQE and UIQM (specialized underwater metrics)
    - Information content (entropy, gradient magnitude)
    """

    def __init__(self, image_path: str | None = None, image_array: np.ndarray | None = None, scale_factor: float = 1.0):
        """
        Initialize with either image path or numpy array.

        Parameters
        ----------
        image_path : str | None
            Path to the image file
        image_array : np.ndarray | None
            Numpy array of the image (BGR format)
        scale_factor : float
            Scaling factor for resizing (e.g., 0.5 for 50% size). Default: 1.0 (no scaling)

        Raises
        ------
        ValueError
            If neither or both parameters are provided, or invalid scale_factor
        """
        if image_path is None and image_array is None:
            raise ValueError("Either image_path or image_array must be provided")
        if image_path is not None and image_array is not None:
            raise ValueError("Only one of image_path or image_array should be provided")

        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")

        if image_path:
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise FileNotFoundError(f"Could not load image from {image_path}")
        else:
            self.image = image_array

        # Apply scaling if scale_factor is not 1.0
        if scale_factor != 1.0:
            h, w = self.image.shape[:2]
            new_size = (int(w * scale_factor), int(h * scale_factor))
            self.image = cv2.resize(self.image, new_size, interpolation=cv2.INTER_AREA)

        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image_lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        self.image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        self.height, self.width = self.image.shape[:2]

    def calculate_all_metrics(self) -> dict[str, float]:
        """
        Calculate all available metrics for the underwater image.

        Returns
        -------
        dict[str, float]
            Dictionary containing all computed metrics
        """
        metrics = {}

        # Color cast metrics (blue water problem)
        metrics.update(self.analyze_color_cast())

        # Contrast metrics
        metrics.update(self.analyze_contrast())

        # Sharpness and blur
        metrics['sharpness_laplacian'] = self.calculate_sharpness_laplacian()
        metrics['sharpness_gradient'] = self.calculate_sharpness_gradient()
        metrics['blur_estimate'] = self.estimate_blur()

        # Feature richness
        metrics.update(self.analyze_feature_richness())

        # Visibility and turbidity
        metrics['visibility_score'] = self.estimate_visibility()
        metrics['turbidity_score'] = self.estimate_turbidity()

        # Underwater-specific metrics
        metrics['uciqe_score'] = self.calculate_uciqe()
        metrics['uiqm_score'] = self.calculate_uiqm()

        # Information content
        metrics['entropy_gray'] = self.calculate_entropy(self.image_gray)
        metrics['entropy_color'] = self.calculate_color_entropy()

        # Color distribution
        metrics.update(self.analyze_color_distribution())

        # Overall quality scores
        metrics['feature_usefulness_score'] = self.calculate_feature_usefulness()
        metrics['marine_science_value'] = self.calculate_marine_science_value()


        # User-registered custom metrics (see register_metric above).
        for _name, _fn in _CUSTOM_METRICS.items():
            metrics[_name] = float(_fn(self))

        return metrics

    def analyze_color_cast(self) -> dict[str, float]:
        """
        Analyze blue-green color cast (blue water problem indicator).

        Returns
        -------
        dict[str, float]
            Dictionary with color cast metrics
        """
        # Calculate mean values for each channel
        b_mean = np.mean(self.image[:, :, 0])
        g_mean = np.mean(self.image[:, :, 1])
        r_mean = np.mean(self.image[:, :, 2])

        total_mean = (b_mean + g_mean + r_mean) / 3

        # Normalized channel differences
        blue_dominance = (b_mean - r_mean) / 255.0
        green_dominance = (g_mean - r_mean) / 255.0
        blue_green_ratio = b_mean / (g_mean + 1e-6)

        # Blue water problem severity (0-1, higher = more severe)
        blue_water_severity = np.clip(blue_dominance + 0.5 * green_dominance, 0, 1)

        # Color temperature (simplified)
        color_temperature = (b_mean + g_mean) / (r_mean + 1e-6)

        # Standard deviation of color channels (uniformity)
        color_std = np.std([b_mean, g_mean, r_mean])

        return {
            'blue_channel_mean': float(b_mean),
            'green_channel_mean': float(g_mean),
            'red_channel_mean': float(r_mean),
            'blue_dominance': float(blue_dominance),
            'green_dominance': float(green_dominance),
            'blue_green_ratio': float(blue_green_ratio),
            'blue_water_severity': float(blue_water_severity),
            'color_temperature': float(color_temperature),
            'color_channel_std': float(color_std),
        }

    def analyze_contrast(self) -> dict[str, float]:
        """
        Analyze various contrast measures.

        Returns
        -------
        dict[str, float]
            Dictionary with contrast metrics
        """
        # Global contrast (RMS contrast)
        rms_contrast = np.std(self.image_gray) / (np.mean(self.image_gray) + 1e-6)

        # Michelson contrast. Cast to float first: image_gray is uint8, so
        # max_val + min_val wraps whenever it exceeds 255, which drove the
        # denominator to 0 and the metric to ~1e8 (bounded [0, 1] in theory).
        max_val = float(np.max(self.image_gray))
        min_val = float(np.min(self.image_gray))
        michelson_contrast = (max_val - min_val) / (max_val + min_val + 1e-6)

        # Local contrast (average of local standard deviations)
        kernel_size = 15
        local_mean = cv2.blur(self.image_gray.astype(float), (kernel_size, kernel_size))
        local_variance = cv2.blur((self.image_gray.astype(float) ** 2), (kernel_size, kernel_size)) - local_mean ** 2
        local_std = np.sqrt(np.maximum(local_variance, 0))
        local_contrast = np.mean(local_std)

        # Weber contrast (using patches)
        patch_size = 32
        weber_contrasts = []
        for i in range(0, self.height - patch_size, patch_size):
            for j in range(0, self.width - patch_size, patch_size):
                patch = self.image_gray[i:i+patch_size, j:j+patch_size]
                center = patch[patch_size//4:3*patch_size//4, patch_size//4:3*patch_size//4]
                surround = patch.copy()
                surround[patch_size//4:3*patch_size//4, patch_size//4:3*patch_size//4] = 0

                center_mean = np.mean(center)
                surround_mean = np.mean(surround[surround > 0]) if np.any(surround > 0) else center_mean

                if surround_mean > 1:
                    weber_contrasts.append(abs(center_mean - surround_mean) / surround_mean)

        weber_contrast = np.mean(weber_contrasts) if weber_contrasts else 0.0

        # Contrast in LAB space
        l_channel = self.image_lab[:, :, 0]
        lab_contrast = np.std(l_channel) / (np.mean(l_channel) + 1e-6)

        return {
            'rms_contrast': float(rms_contrast),
            'michelson_contrast': float(michelson_contrast),
            'local_contrast': float(local_contrast),
            'weber_contrast': float(weber_contrast),
            'lab_contrast': float(lab_contrast),
        }

    def calculate_sharpness_laplacian(self) -> float:
        """
        Calculate sharpness using Laplacian variance.

        Returns
        -------
        float
            Sharpness score (higher = sharper)
        """
        laplacian = cv2.Laplacian(self.image_gray, cv2.CV_64F)
        return float(np.var(laplacian))

    def calculate_sharpness_gradient(self) -> float:
        """
        Calculate sharpness using gradient magnitude.

        Returns
        -------
        float
            Gradient-based sharpness score
        """
        sobelx = cv2.Sobel(self.image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.image_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        return float(np.mean(gradient_magnitude))

    def estimate_blur(self) -> float:
        """
        Estimate blur using frequency domain analysis.

        Returns
        -------
        float
            Blur estimate (higher = more blurry)
        """
        # Use FFT to analyze frequency content
        f_transform = np.fft.fft2(self.image_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # Calculate the ratio of high frequencies to low frequencies
        cy, cx = self.height // 2, self.width // 2
        radius = min(cy, cx) // 4

        # High frequency region (outer ring)
        y, x = np.ogrid[:self.height, :self.width]
        mask_high = ((x - cx)**2 + (y - cy)**2) >= radius**2

        # Low frequency region (center circle)
        mask_low = ((x - cx)**2 + (y - cy)**2) < radius**2

        high_freq_energy = np.sum(magnitude_spectrum[mask_high])
        low_freq_energy = np.sum(magnitude_spectrum[mask_low])

        # Blur estimate: lower ratio = more blur
        blur_ratio = low_freq_energy / (high_freq_energy + 1e-6)

        # Normalize to 0-1 range (higher = more blurry)
        blur_estimate = np.tanh(blur_ratio / 100)

        return float(blur_estimate)

    def analyze_feature_richness(self) -> dict[str, float]:
        """
        Analyze the richness of features in the image.

        Returns
        -------
        dict[str, float]
            Dictionary with feature richness metrics
        """
        # Edge density
        edges = cv2.Canny(self.image_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (self.height * self.width)

        # Multi-scale edge detection
        edge_densities = []
        for sigma in [1.0, 2.0, 4.0]:
            blurred = cv2.GaussianBlur(self.image_gray, (0, 0), sigma)
            edges_scale = cv2.Canny(blurred, 50, 150)
            edge_densities.append(np.sum(edges_scale > 0) / (self.height * self.width))

        multi_scale_edge_density = np.mean(edge_densities)

        # Corner detection (Harris corners)
        corners = cv2.cornerHarris(self.image_gray.astype(np.float32), blockSize=2, ksize=3, k=0.04)
        corner_density = np.sum(corners > 0.01 * corners.max()) / (self.height * self.width)

        # FAST keypoints
        fast = cv2.FastFeatureDetector_create(threshold=20)
        keypoints = fast.detect(self.image, None)
        keypoint_density = len(keypoints) / (self.height * self.width) * 1000  # per 1000 pixels

        # Texture complexity (using Gray Level Co-occurrence Matrix approximation)
        texture_score = self._calculate_texture_complexity()

        return {
            'edge_density': float(edge_density),
            'multi_scale_edge_density': float(multi_scale_edge_density),
            'corner_density': float(corner_density),
            'keypoint_density': float(keypoint_density),
            'texture_complexity': float(texture_score),
        }

    def _calculate_texture_complexity(self) -> float:
        """
        Calculate texture complexity using local binary patterns approximation.

        Returns
        -------
        float
            Texture complexity score
        """
        # Simplified texture analysis using local variance
        kernel_size = 9
        local_mean = cv2.blur(self.image_gray.astype(float), (kernel_size, kernel_size))
        local_variance = cv2.blur((self.image_gray.astype(float) ** 2), (kernel_size, kernel_size)) - local_mean ** 2

        texture_score = np.mean(np.sqrt(np.maximum(local_variance, 0)))
        return float(texture_score)

    def estimate_visibility(self) -> float:
        """
        Estimate underwater visibility based on multiple factors.

        Returns
        -------
        float
            Visibility score (0-1, higher = better visibility)
        """
        # Combine multiple indicators
        contrast_score = self.analyze_contrast()['rms_contrast']
        sharpness_score = self.calculate_sharpness_gradient()
        edge_score = self.analyze_feature_richness()['edge_density']

        # Normalize scores
        contrast_norm = np.clip(contrast_score / 0.5, 0, 1)
        sharpness_norm = np.clip(sharpness_score / 30, 0, 1)
        edge_norm = np.clip(edge_score / 0.2, 0, 1)

        # Weighted combination
        visibility = 0.4 * contrast_norm + 0.4 * sharpness_norm + 0.2 * edge_norm

        return float(visibility)

    def estimate_turbidity(self) -> float:
        """
        Estimate water turbidity (0-1, higher = more turbid).

        Returns
        -------
        float
            Turbidity estimate
        """
        # Turbidity indicators:
        # 1. High blue-green dominance
        # 2. Low contrast
        # 3. High blur
        # 4. Low feature density

        color_cast = self.analyze_color_cast()
        blue_water = color_cast['blue_water_severity']

        contrast = self.analyze_contrast()['rms_contrast']
        contrast_inv = 1 - np.clip(contrast / 0.5, 0, 1)

        blur = self.estimate_blur()

        edge_density = self.analyze_feature_richness()['edge_density']
        edge_inv = 1 - np.clip(edge_density / 0.2, 0, 1)

        # Combine indicators
        turbidity = 0.3 * blue_water + 0.3 * contrast_inv + 0.2 * blur + 0.2 * edge_inv

        return float(turbidity)

    def calculate_uciqe(self) -> float:
        """
        Calculate UCIQE (Underwater Color Image Quality Evaluation).

        Reference: Yang, M., & Sowmya, A. (2015). "An underwater color image
        quality evaluation metric." IEEE TIP.

        Note: releases up to v1.0.1 omitted the +128 offset that OpenCV applies
        to the a and b channels of 8-bit LAB, which inflated this metric by
        roughly 5x. Values reported in the UNIQAT manuscript were produced with
        that earlier definition and are not comparable to values from this
        version. A model trained on pre-v1.0.2 labels predicts the old variant.

        Returns
        -------
        float
            UCIQE score
        """
        # Convert to LAB. OpenCV stores 8-bit LAB with a and b offset by +128,
        # so neutral grey is (128, 128) rather than (0, 0). The offset must be
        # removed before computing chroma, otherwise a neutral grey image scores
        # chroma 0.71 instead of 0 and UCIQE comes out roughly 5x inflated.
        lab = self.image_lab.astype(float) / 255.0
        a_star = lab[:, :, 1] - 128.0 / 255.0
        b_star = lab[:, :, 2] - 128.0 / 255.0

        # Chroma
        chroma = np.sqrt(a_star**2 + b_star**2)

        # Standard deviation of chroma
        sigma_c = np.std(chroma)

        # Contrast of luminance
        l_channel = lab[:, :, 0]
        con_l = np.std(l_channel)

        # Average saturation
        sat = chroma / (l_channel + 1e-6)
        mu_s = np.mean(sat)

        # UCIQE formula
        c1, c2, c3 = 0.4680, 0.2745, 0.2576
        uciqe = c1 * sigma_c + c2 * con_l + c3 * mu_s

        return float(uciqe)

    def calculate_uiqm(self) -> float:
        """
        Calculate UIQM (Underwater Image Quality Measure).

        Reference: Panetta, K., Gao, C., & Agaian, S. (2016).
        "Human-Visual-System-Inspired Underwater Image Quality Measures."
        IEEE JOE.

        Returns
        -------
        float
            UIQM score
        """
        # UIQM components: UICM (colorfulness), UISM (sharpness), UIConM (contrast)

        # UICM - Underwater Image Colorfulness Measure
        rg = self.image_rgb[:, :, 0].astype(float) - self.image_rgb[:, :, 1].astype(float)
        yb = (self.image_rgb[:, :, 0].astype(float) + self.image_rgb[:, :, 1].astype(float)) / 2 - self.image_rgb[:, :, 2].astype(float)

        rg_mean = np.mean(rg)
        yb_mean = np.mean(yb)
        rg_std = np.std(rg)
        yb_std = np.std(yb)

        uicm = -0.0268 * np.sqrt(rg_mean**2 + yb_mean**2) + 0.1586 * np.sqrt(rg_std**2 + yb_std**2)

        # UISM - Underwater Image Sharpness Measure
        # Use edge detection with Sobel
        sobelx = cv2.Sobel(self.image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.image_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)

        # Calculate EME (Enhancement Measure Estimation)
        uism = self._calculate_eme(gradient)

        # UIConM - Underwater Image Contrast Measure
        uiconm = self._calculate_uiconm()

        # Combine components
        c1, c2, c3 = 0.0282, 0.2953, 3.5753
        uiqm = c1 * uicm + c2 * uism + c3 * uiconm

        return float(uiqm)

    def _calculate_eme(self, image: np.ndarray, block_size: int = 8) -> float:
        """Calculate EME (Enhancement Measure Estimation)."""
        height, width = image.shape
        eme = 0.0
        count = 0

        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = image[i:min(i+block_size, height), j:min(j+block_size, width)]
                if block.size == 0:
                    continue

                max_val = np.max(block)
                min_val = np.min(block)

                if min_val > 0:
                    eme += np.log(max_val / min_val + 1e-6)
                    count += 1

        return eme / (count + 1e-6)

    def _calculate_uiconm(self) -> float:
        """Calculate UIConM (Underwater Image Contrast Measure)."""
        # Use LOG filter for contrast
        log_amee = 0.0

        for channel in range(3):
            channel_data = self.image[:, :, channel].astype(float)
            log_filtered = ndimage.gaussian_laplace(channel_data, sigma=1)
            log_amee += self._calculate_eme(np.abs(log_filtered))

        return log_amee / 3.0

    def calculate_entropy(self, image: np.ndarray) -> float:
        """
        Calculate Shannon entropy of an image.

        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale)

        Returns
        -------
        float
            Entropy value
        """
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-6)
        hist = hist[hist > 0]

        return float(-np.sum(hist * np.log2(hist)))

    def calculate_color_entropy(self) -> float:
        """
        Calculate entropy across color channels.

        Returns
        -------
        float
            Average color entropy
        """
        entropies = []
        for channel in range(3):
            entropies.append(self.calculate_entropy(self.image[:, :, channel]))

        return float(np.mean(entropies))

    def analyze_color_distribution(self) -> dict[str, float]:
        """
        Analyze color distribution characteristics.

        Returns
        -------
        dict[str, float]
            Dictionary with color distribution metrics
        """
        # Histogram uniformity
        hist_b = cv2.calcHist([self.image], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([self.image], [1], None, [256], [0, 256]).flatten()
        hist_r = cv2.calcHist([self.image], [2], None, [256], [0, 256]).flatten()

        # Normalize histograms
        hist_b = hist_b / (hist_b.sum() + 1e-6)
        hist_g = hist_g / (hist_g.sum() + 1e-6)
        hist_r = hist_r / (hist_r.sum() + 1e-6)

        # Chi-square distance from uniform distribution
        uniform = np.ones(256) / 256

        def chi_square(hist1, hist2):
            return np.sum((hist1 - hist2)**2 / (hist2 + 1e-6))

        uniformity_b = chi_square(hist_b, uniform)
        uniformity_g = chi_square(hist_g, uniform)
        uniformity_r = chi_square(hist_r, uniform)

        # Dynamic range utilization
        def dynamic_range(hist):
            # Percentage of bins used
            return np.sum(hist > 0.001) / 256

        dynamic_range_b = dynamic_range(hist_b)
        dynamic_range_g = dynamic_range(hist_g)
        dynamic_range_r = dynamic_range(hist_r)

        # Color diversity (using HSV)
        hue_hist = cv2.calcHist([self.image_hsv], [0], None, [180], [0, 180]).flatten()
        hue_hist = hue_hist / (hue_hist.sum() + 1e-6)
        hue_diversity = self.calculate_entropy(self.image_hsv[:, :, 0])

        return {
            'histogram_uniformity_b': float(uniformity_b),
            'histogram_uniformity_g': float(uniformity_g),
            'histogram_uniformity_r': float(uniformity_r),
            'dynamic_range_b': float(dynamic_range_b),
            'dynamic_range_g': float(dynamic_range_g),
            'dynamic_range_r': float(dynamic_range_r),
            'hue_diversity': float(hue_diversity),
        }

    def calculate_feature_usefulness(self) -> float:
        """
        Calculate overall feature usefulness for computer vision tasks (0-100).

        This score indicates how useful the image is for feature extraction,
        object detection, and other CV tasks.

        Returns
        -------
        float
            Feature usefulness score (0-100)
        """
        # Get key metrics
        features = self.analyze_feature_richness()
        contrast = self.analyze_contrast()['rms_contrast']
        sharpness = self.calculate_sharpness_gradient()
        entropy = self.calculate_entropy(self.image_gray)
        turbidity = self.estimate_turbidity()

        # Normalize and weight components
        edge_score = np.clip(features['edge_density'] / 0.2, 0, 1) * 25
        keypoint_score = np.clip(features['keypoint_density'] / 2.0, 0, 1) * 20
        contrast_score = np.clip(contrast / 0.5, 0, 1) * 20
        sharpness_score = np.clip(sharpness / 30, 0, 1) * 15
        entropy_score = np.clip(entropy / 7.5, 0, 1) * 10
        clarity_score = (1 - turbidity) * 10

        total_score = (edge_score + keypoint_score + contrast_score +
                      sharpness_score + entropy_score + clarity_score)

        return float(total_score)

    def calculate_marine_science_value(self) -> float:
        """
        Calculate value for marine science applications (0-100).

        This score considers visibility, color fidelity, feature clarity,
        and freedom from blue water problem artifacts.

        Returns
        -------
        float
            Marine science value score (0-100)
        """
        # Get key metrics
        visibility = self.estimate_visibility()
        turbidity = self.estimate_turbidity()
        color_cast = self.analyze_color_cast()
        features = self.analyze_feature_richness()
        uciqe = self.calculate_uciqe()

        # Component scores
        visibility_score = visibility * 30
        clarity_score = (1 - turbidity) * 25
        color_quality_score = (1 - color_cast['blue_water_severity']) * 20
        feature_score = np.clip(features['edge_density'] / 0.2, 0, 1) * 15
        underwater_quality_score = np.clip(uciqe / 0.6, 0, 1) * 10

        total_score = (visibility_score + clarity_score + color_quality_score +
                      feature_score + underwater_quality_score)

        return float(total_score)
