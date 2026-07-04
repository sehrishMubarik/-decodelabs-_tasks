"""
DecodeLabs - Project 4: Image or Text Recognition (Basic)
PATH 1 — Optical Character Recognition (OCR)
------------------------------------------------------------
Goal: Implement a basic text-recognition task using a pre-trained
library (pytesseract / Google's Tesseract engine), following the
"Logic Skeleton" pre-processing pipeline from the slide deck, and
gate the final output behind an 80% confidence threshold.

Pipeline (matches the deck exactly):

    INPUT   -> Load raw image (a 3D array: Height x Width x Color)
    PROCESS -> Step 1: Grayscale conversion  (collapse RGB -> 1D intensity)
               Step 2: Gaussian Blur          (remove noise/artifacts)
               Step 3: Adaptive/Otsu Thresholding (force binary black/white)
               Step 4: Deskew                 (straighten tilted text lines)
               Step 5: Run pytesseract with a tuned PSM (Page Segmentation Mode)
    OUTPUT  -> Filter every recognized word by the 80% confidence gate,
               then display the clean, validated text string

Milestone Validation Gates (from the deck):
    1. Library Integration      -> pytesseract, wired up cleanly
    2. Pre-Processing Integrity -> grayscale + adaptive thresholding shown
    3. Accuracy Benchmarking    -> 80% minimum confidence enforced
    4. Visual Confirmation      -> legible final text output
"""

import cv2
import numpy as np
import pytesseract

CONFIDENCE_THRESHOLD = 80  # The "80% Gate" — Project 4's minimum standard


# ---------------------------------------------------------------------
# PHASE 1: INPUT
# ---------------------------------------------------------------------
def load_image(path):
    """An image, to a machine, isn't a picture — it's a 3D array of
    Height x Width x Color-Channel intensity values (0-255)."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {path}")
    return image


# ---------------------------------------------------------------------
# PHASE 2: PROCESS — The Logic Skeleton (Pre-Processing)
# ---------------------------------------------------------------------
def to_grayscale(image):
    """Step 1: Collapse the 3D RGB matrix into a 1D intensity matrix.
    Color is noise for text recognition — only shape/contrast matters."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray_image, kernel_size=(5, 5)):
    """Step 2: Smooth out micro-imperfections and sensor noise before
    we force a hard black/white decision."""
    return cv2.GaussianBlur(gray_image, kernel_size, 0)


def apply_adaptive_threshold(blurred_image):
    """Step 3: Otsu's Method — force every pixel to choose a side.
    Automatically calculates the optimal cutoff intensity instead of
    hardcoding one, then converts grayscale into pure black-and-white."""
    _, binary = cv2.threshold(
        blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def deskew(binary_image):
    """Step 4: Detect the dominant text angle and rotate the image back
    to a perfect horizontal baseline. Uses the minimum-area bounding
    rectangle of all 'ink' pixels to estimate rotation."""
    coords = np.column_stack(np.where(binary_image < 255))
    if len(coords) == 0:
        return binary_image  # nothing to deskew

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = binary_image.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary_image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess_pipeline(image):
    """Run the full Logic Skeleton: Grayscale -> Blur -> Threshold -> Deskew."""
    gray = to_grayscale(image)
    blurred = apply_gaussian_blur(gray)
    binary = apply_adaptive_threshold(blurred)
    deskewed = deskew(binary)
    return deskewed


# ---------------------------------------------------------------------
# PHASE 3: OUTPUT — Recognition + Confidence Gate
# ---------------------------------------------------------------------
def recognize_text(processed_image, psm=3):
    """Run pytesseract with a tuned Page Segmentation Mode and return
    word-level results including per-word confidence scores.

    PSM cheat sheet (from the deck):
      3  -> fully automatic (default, varied layouts)
      6  -> single uniform block of text (book pages)
      7  -> single text line (number plates/headers)
      11 -> sparse, scattered text (invoices)
    """
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(
        processed_image, config=config, output_type=pytesseract.Output.DICT
    )
    return data


def apply_confidence_gate(data, threshold=CONFIDENCE_THRESHOLD):
    """The 80% Gate: only trust words the model is confident about.
    Below the threshold, the AI's own guess is too uncertain to
    surface as a validated result."""
    accepted_words = []
    for word, conf in zip(data["text"], data["conf"]):
        conf = float(conf)
        if word.strip() and conf >= threshold:
            accepted_words.append((word, conf))
    return accepted_words


def run_ocr_pipeline(image_path, psm=3, threshold=CONFIDENCE_THRESHOLD):
    print("=" * 60)
    print("PATH 1: Optical Character Recognition (pytesseract)")
    print("=" * 60)

    image = load_image(image_path)
    print(f"Loaded image: {image_path}  (shape: {image.shape})")

    processed = preprocess_pipeline(image)
    cv2.imwrite("ocr_preprocessed.png", processed)
    print("Pre-processing complete: grayscale -> blur -> Otsu threshold -> deskew")
    print("(saved to ocr_preprocessed.png for visual confirmation)\n")

    data = recognize_text(processed, psm=psm)
    accepted = apply_confidence_gate(data, threshold)

    print(f"Confidence gate: >= {threshold}%\n")
    if accepted:
        print("VALIDATED OUTPUT (Visual Confirmation):")
        full_text = " ".join(word for word, _ in accepted)
        print(f'  "{full_text}"\n')
        print("Per-word confidence:")
        for word, conf in accepted:
            print(f"  {word:20s} {conf:.1f}%")
        avg_conf = sum(c for _, c in accepted) / len(accepted)
        print(f"\nAverage validated confidence: {avg_conf:.1f}%")
        print(f"Milestone check: {'PASSED' if avg_conf >= threshold else 'FAILED'} "
              f"(needs >= {threshold}%)")
    else:
        print("No text passed the 80% confidence gate. Try a clearer "
              "image or a different --psm mode.")


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample_text.png"
    run_ocr_pipeline(image_path, psm=3)
