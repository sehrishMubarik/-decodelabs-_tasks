"""
DecodeLabs - Project 4: Image or Text Recognition (Basic)
PATH 2 — Object Detection with MobileNet-SSD
------------------------------------------------------------
Goal: Implement basic object recognition using a pre-trained deep
learning model (Transfer Learning), following the deck's pipeline:
Blob Construction -> Single Shot Detection -> Bounding Box Decoding
-> the 80% Confidence Gate.

Pipeline (matches the deck exactly):

    INPUT   -> Load raw image (Height x Width x 3 color-channel array)
    PROCESS -> Step 1: Blob Construction (cv2.dnn.blobFromImage) —
                        mean subtraction + resize to the network's
                        required 300x300 input dimensions
               Step 2: Single Shot Detection forward pass through
                        MobileNet-SSD (pre-trained on ImageNet, then
                        transfer-learned onto VOC object classes)
               Step 3: Decode normalized bounding-box coordinates and
                        scale them back to the original image's pixels
    OUTPUT  -> Apply the 80% confidence gate (Softmax-derived score),
               draw + label only the boxes that pass, save the result

Milestone Validation Gates (from the deck):
    1. Library Integration      -> cv2.dnn, wired up cleanly
    2. Pre-Processing Integrity -> blob construction shown explicitly
    3. Accuracy Benchmarking    -> 80% minimum confidence enforced
    4. Visual Confirmation      -> accurate bounding boxes with labels
"""

import cv2
import numpy as np

CONFIDENCE_THRESHOLD = 0.80  # The "80% Gate" — Project 4's minimum standard

# MobileNet-SSD (VOC-trained) recognizes these 21 classes, including background
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

# A distinct color per class for the bounding-box overlay
np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# ---------------------------------------------------------------------
# PHASE 1: INPUT
# ---------------------------------------------------------------------
def load_model(prototxt_path, model_path):
    """Load the pre-trained MobileNet-SSD architecture (prototxt) and
    its learned weights (caffemodel). This is Transfer Learning: we
    didn't train this — it already understands millions of images."""
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {path}")
    return image


# ---------------------------------------------------------------------
# PHASE 2: PROCESS — Blob Construction + Forward Pass
# ---------------------------------------------------------------------
def build_blob(image, size=(300, 300), scale=0.007843, mean=127.5):
    """Convert the raw image into the 4D 'blob' the network expects:
    resize to 300x300, subtract the mean pixel value, and scale —
    exactly the preprocessing MobileNet-SSD was trained on."""
    blob = cv2.dnn.blobFromImage(image, scale, size, (mean, mean, mean))
    return blob


def run_detection(net, blob):
    """Single Shot Detector: one forward pass through the network
    returns every candidate bounding box AND its class confidence
    simultaneously (vs. older multi-pass sliding-window approaches)."""
    net.setInput(blob)
    detections = net.forward()
    return detections


# ---------------------------------------------------------------------
# PHASE 3: OUTPUT — Confidence Gate + Bounding Box Decoding
# ---------------------------------------------------------------------
def decode_and_draw(image, detections, threshold=CONFIDENCE_THRESHOLD):
    """For every detection: apply the 80% confidence gate, then
    translate normalized (0-1) coordinates into real pixel coordinates
    and draw a labeled bounding box."""
    (h, w) = image.shape[:2]
    accepted = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # THE 80% GATE — drop anything the model isn't confident about
        if confidence < threshold:
            continue

        class_id = int(detections[0, 0, i, 1])
        label_name = CLASSES[class_id] if class_id < len(CLASSES) else "unknown"

        # Decode normalized coords -> real pixel coordinates
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype("int")

        color = COLORS[class_id % len(COLORS)]
        label = f"{label_name}: {confidence * 100:.1f}%"
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 2)
        y_text = start_y - 10 if start_y - 10 > 10 else start_y + 20
        cv2.putText(image, label, (start_x, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        accepted.append((label_name, confidence, (start_x, start_y, end_x, end_y)))

    return image, accepted


def run_detection_pipeline(image_path, prototxt_path, model_path,
                            threshold=CONFIDENCE_THRESHOLD,
                            output_path="detection_output.png"):
    print("=" * 60)
    print("PATH 2: Object Detection (OpenCV DNN + MobileNet-SSD)")
    print("=" * 60)

    net = load_model(prototxt_path, model_path)
    image = load_image(image_path)
    print(f"Loaded image: {image_path}  (shape: {image.shape})")

    blob = build_blob(image)
    print(f"Blob constructed: shape {blob.shape} (mean-subtracted, 300x300)")

    detections = run_detection(net, blob)
    output_image, accepted = decode_and_draw(image.copy(), detections, threshold)

    cv2.imwrite(output_path, output_image)
    print(f"\nConfidence gate: >= {threshold:.0%}\n")

    if accepted:
        print("VALIDATED DETECTIONS (Visual Confirmation):")
        for label, conf, box in accepted:
            print(f"  {label:12s}  confidence={conf:.1%}  box={box}")
        print(f"\n(Annotated image saved to {output_path})")
        print("Milestone check: PASSED")
    else:
        print("No detections passed the 80% confidence gate.")
        print("Milestone check: FAILED — try a clearer image with a "
              "recognizable object (person, car, dog, etc.)")


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample_object.png"
    run_detection_pipeline(
        image_path,
        prototxt_path="model/deploy.prototxt",
        model_path="model/MobileNetSSD_deploy.caffemodel",
    )
