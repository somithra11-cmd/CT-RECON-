import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, median_filter
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="CT-RECON",
    page_icon="🩻",
    layout="wide"
)

st.title("CT-RECON")
st.header("CT Image Reconstruction and Noise Reduction System")
st.write(
    "A software-based demonstration of Back Projection, "
    "Filtered Back Projection and image denoising."
)

st.info(
    "Educational simulation only. Uploaded CT slices are used to generate "
    "simulated projections. This is not a clinical diagnostic system."
)


# ------------------------------------------------------------
# 2. IMAGE PREPROCESSING FUNCTIONS
# ------------------------------------------------------------
IMAGE_SIZE = 256


def normalize_image(image):
    """Convert an image to a finite floating-point image in the range 0–1."""
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    if high <= low:
        low = float(np.min(image))
        high = float(np.max(image))

    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return image.astype(np.float32)


def fit_image_to_square(image, size=IMAGE_SIZE):
    """
    Resize an image while preserving its aspect ratio, then pad it to a
    square. The square image is suitable for the circular Radon transform.
    """
    image = np.asarray(image, dtype=np.float32)

    if image.ndim == 3:
        # Convert RGB/RGBA images to grayscale.
        image = image[..., :3]
        image = rgb2gray(image)

    if image.ndim != 2:
        raise ValueError("Please upload a 2D grayscale CT slice or a standard RGB image.")

    image = normalize_image(image)

    height, width = image.shape
    if height == 0 or width == 0:
        raise ValueError("The uploaded image is empty.")

    scale = min(size / height, size / width)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))

    resized = resize(
        image,
        (new_height, new_width),
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.float32)

    square = np.zeros((size, size), dtype=np.float32)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    square[top:top + new_height, left:left + new_width] = resized

    return square


def load_uploaded_image(uploaded_file):
    """Load PNG/JPG/TIFF images and DICOM files."""
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if filename.endswith(".dcm"):
        try:
            import pydicom
        except ImportError:
            raise ImportError(
                "DICOM support needs pydicom. In the VS Code terminal run: "
                "py -3.12 -m pip install pydicom"
            )

        dicom_dataset = pydicom.dcmread(io.BytesIO(file_bytes))
        image = dicom_dataset.pixel_array.astype(np.float32)

        # For multi-frame DICOM, use the first frame in this 2D project.
        if image.ndim == 3 and image.shape[0] > 1 and image.shape[-1] not in (3, 4):
            image = image[0]

        # Convert stored pixel values to modality values, commonly HU.
        slope = float(getattr(dicom_dataset, "RescaleSlope", 1))
        intercept = float(getattr(dicom_dataset, "RescaleIntercept", 0))
        image = image * slope + intercept

        # MONOCHROME1 means low values are displayed as white.
        if getattr(dicom_dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
            image = np.max(image) + np.min(image) - image

        return image

    image = imread(io.BytesIO(file_bytes))
    return image


# ------------------------------------------------------------
# 3. INPUT SELECTION: PHANTOM OR UPLOADED CT IMAGE
# ------------------------------------------------------------
st.subheader("1. Select Input Image")

input_mode = st.radio(
    "Choose the image source:",
    ["Shepp-Logan Phantom", "Upload Real CT Image"],
    horizontal=True
)

if input_mode == "Shepp-Logan Phantom":
    original_image = fit_image_to_square(shepp_logan_phantom())
    st.success("Shepp–Logan Phantom selected.")
    st.caption(
        "The Shepp–Logan phantom is a mathematical test image, "
        "not an image of a real human body part."
    )

else:
    uploaded_file = st.file_uploader(
        "Upload a 2D CT image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "dcm"],
        help=(
            "Supported formats: PNG, JPG, JPEG, TIFF and DICOM (.dcm). "
            "For ordinary image files, use a grayscale CT slice where possible."
        )
    )

    if uploaded_file is None:
        st.warning("Upload a CT image to continue. The reconstruction will appear here.")
        st.stop()

    try:
        loaded_image = load_uploaded_image(uploaded_file)
        original_image = fit_image_to_square(loaded_image)
        st.success(f"Image loaded successfully: {uploaded_file.name}")
        st.caption(
            "The uploaded image has been converted to grayscale, normalized "
            "and fitted into a 256 × 256 square while preserving its aspect ratio."
        )
    except Exception as error:
        st.error(f"Could not load this image: {error}")
        st.stop()


# ------------------------------------------------------------
# 4. USER CONTROLS
# ------------------------------------------------------------
st.subheader("2. Reconstruction Settings")

control_col1, control_col2 = st.columns(2)

with control_col1:
    number_of_angles = st.slider(
        "Number of projection angles",
        min_value=30,
        max_value=180,
        value=90,
        step=30
    )

with control_col2:
    noise_percent = st.slider(
        "Gaussian noise level (%)",
        min_value=0,
        max_value=30,
        value=5,
        step=1
    )

st.caption(
    "Increasing the number of projection angles provides more projection views. "
    "Increasing the noise level adds more simulated Gaussian noise to the sinogram."
)


# ------------------------------------------------------------
# 5. RADON TRANSFORM AND SINOGRAM GENERATION
# ------------------------------------------------------------
st.subheader("3. CT Projection Generation")

theta = np.linspace(0.0, 180.0, number_of_angles, endpoint=False)

sinogram = radon(
    original_image,
    theta=theta,
    circle=True
)

# Add Gaussian noise to the projection data (sinogram).
rng = np.random.default_rng(42)
noise_standard_deviation = (noise_percent / 100.0) * float(np.std(sinogram))

if noise_percent > 0:
    noisy_sinogram = sinogram + rng.normal(
        loc=0.0,
        scale=noise_standard_deviation,
        size=sinogram.shape
    )
else:
    noisy_sinogram = sinogram.copy()


# # ------------------------------------------------------------
# 6. CLEAN AND NOISY RECONSTRUCTION
# ------------------------------------------------------------

st.subheader("4. Image Reconstruction")

# Clean reconstruction (reference)
clean_fbp = iradon(
    sinogram,
    theta=theta,
    filter_name="ramp",
    circle=True,
    output_size=IMAGE_SIZE
)

# Reconstruction from noisy projections
back_projection = iradon(
    noisy_sinogram,
    theta=theta,
    filter_name=None,
    circle=True,
    output_size=IMAGE_SIZE
)

noisy_fbp = iradon(
    noisy_sinogram,
    theta=theta,
    filter_name="ramp",
    circle=True,
    output_size=IMAGE_SIZE
)

# # ------------------------------------------------------------
# 7. NOISE REDUCTION
# ------------------------------------------------------------

st.subheader("5. Noise Reduction")

gaussian_result = gaussian_filter(
    noisy_fbp,
    sigma=1
)

median_result = median_filter(
    noisy_fbp,
    size=3
)


# ------------------------------------------------------------
# 8. DISPLAY HELPER
# ------------------------------------------------------------
def display_image(image, title, cmap="gray"):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# # ------------------------------------------------------------
# 9. RECONSTRUCTION RESULTS
# ------------------------------------------------------------

st.subheader("6. Reconstruction Results")

row1 = st.columns(3)

with row1[0]:
    display_image(original_image, "Original Input CT")

with row1[1]:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(noisy_sinogram, cmap="gray", aspect="auto")
    ax.set_title("Noisy Sinogram")
    ax.set_xlabel("Projection Angle")
    ax.set_ylabel("Detector Position")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with row1[2]:
    display_image(back_projection, "Back Projection (BP)")

row2 = st.columns(3)

with row2[0]:
    display_image(clean_fbp, "Clean FBP Reference")

with row2[1]:
    display_image(noisy_fbp, "Noisy FBP")

with row2[2]:
    display_image(gaussian_result, "Gaussian Denoising")

row3 = st.columns(1)

with row3[0]:
    display_image(median_result, "Median Denoising")

# # ------------------------------------------------------------
# 10. IMAGE QUALITY ASSESSMENT
# ------------------------------------------------------------

st.subheader("7. Image Quality Assessment")

def calculate_metrics(reference, reconstructed):

    data_range = float(np.max(reference) - np.min(reference))

    if data_range == 0:
        data_range = 1.0

    mse_value = mean_squared_error(
        reference,
        reconstructed
    )

    psnr_value = peak_signal_noise_ratio(
        reference,
        reconstructed,
        data_range=data_range
    )

    ssim_value = structural_similarity(
        reference,
        reconstructed,
        data_range=data_range
    )

    return {
        "MSE": float(mse_value),
        "PSNR (dB)": float(psnr_value),
        "SSIM": float(ssim_value)
    }

results = {
    "Noisy FBP": calculate_metrics(clean_fbp, noisy_fbp),
    "Gaussian Filter": calculate_metrics(clean_fbp, gaussian_result),
    "Median Filter": calculate_metrics(clean_fbp, median_result)
}

st.write(
    "Comparison against the clean simulated FBP reference."
)

metric_rows = []

for method, values in results.items():
    metric_rows.append({
        "Method": method,
        "MSE": round(values["MSE"], 5),
        "PSNR (dB)": round(values["PSNR (dB)"], 2),
        "SSIM": round(values["SSIM"], 4)
    })

st.dataframe(
    metric_rows,
    width="stretch",
    hide_index=True
)

# ------------------------------------------------------------
# 11. FINAL SUMMARY
# ------------------------------------------------------------
st.subheader("8. Project Summary")

st.success("CT reconstruction and noise reduction simulation completed successfully!")

st.write("The system demonstrates:")
st.markdown(
    """
    1. Shepp–Logan phantom generation or real CT image upload.
    2. Image preprocessing and grayscale conversion.
    3. CT projection generation using the Radon Transform.
    4. Sinogram visualization and Gaussian noise simulation.
    5. Back Projection reconstruction.
    6. Filtered Back Projection reconstruction.
    7. Gaussian and Median noise reduction.
    8. Image quality evaluation using MSE, PSNR and SSIM.
    """
)

st.warning(
    "Research and educational use only. This application does not provide "
    "medical diagnosis and must not be used to make clinical decisions."
)