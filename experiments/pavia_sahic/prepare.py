"""Prepare Pavia University inputs and independent NumPy references for six SAHIC stages."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from scipy.special import softmax

HEIGHT, WIDTH, CLASSES = 610, 340, 9
PIXELS = HEIGHT * WIDTH
MU, TAU = 1.0, 0.02


def probabilities(image, labels):
    pixels = image.reshape(PIXELS, -1).astype(np.float64)
    pixels /= pixels.max()
    rng = np.random.default_rng(20261124)
    labels = labels.ravel()
    chosen = []
    for label in range(1, CLASSES + 1):
        candidates = np.flatnonzero(labels == label)
        chosen.extend(rng.choice(candidates, max(1, len(candidates) // 100), replace=False))
    chosen = np.sort(chosen).astype(np.int32)
    targets = np.eye(CLASSES)[labels[chosen] - 1]
    dictionary = pixels[chosen]
    distance = cdist(dictionary, dictionary, "sqeuclidean")
    gamma = 1 / np.median(distance[distance > 0])
    features = np.exp(-gamma * distance)
    # Proximal gradient with acceleration for the L1 multinomial objective.
    step = 1 / (0.5 * np.linalg.norm(features, 2) ** 2 / len(chosen))
    coefficients = accelerated = np.zeros_like(targets)
    for i in range(1, 151):
        gradient = features.T @ (softmax(features @ accelerated, axis=1) - targets) / len(chosen)
        point = accelerated - step * gradient
        updated = np.sign(point) * np.maximum(np.abs(point) - step * 0.00005, 0)
        accelerated = updated + (i - 1) / (i + 2) * (updated - coefficients)
        coefficients = updated
    result = np.empty((CLASSES, PIXELS))
    for start in range(0, PIXELS, 8192):
        end = min(start + 8192, PIXELS)
        tile = np.exp(-gamma * cdist(pixels[start:end], dictionary, "sqeuclidean"))
        result[:, start:end] = softmax(tile @ coefficients, axis=1).T
    return result, chosen, targets.T


def checkpoint(posterior, chosen, targets, save):
    values = [posterior.copy() for _ in range(4)]
    multipliers = [np.zeros_like(posterior) for _ in range(4)]
    # FFT grids are stored in transposed spatial order, as in the V4 tile.
    shape = (CLASSES, WIDTH, HEIGHT)
    previous_gradient = np.zeros((2, *shape))
    old_dual = np.zeros_like(previous_gradient)
    h = 1 - np.exp(2j * np.pi * np.arange(HEIGHT) / HEIGHT)[None, :]
    v = 1 - np.exp(2j * np.pi * np.arange(WIDTH) / WIDTH)[:, None]
    filters = np.broadcast_arrays(h, v)
    denominator = 1 + np.abs(h)**2 + np.abs(v)**2

    def to_fft_layout(array):
        return array.reshape(CLASSES, HEIGHT, WIDTH).transpose(0, 2, 1)

    for t in range(3):
        compact = sum(a + b for a, b in zip(values, multipliers)) * 0.25
        q = compact.copy()
        q[:, chosen] = targets
        s = q - multipliers[2]
        sums = s.sum(axis=0)
        bias = (1 - sums) / CLASSES
        v1 = (posterior + MU * (q - multipliers[0])) / (1 + MU)
        v2 = np.maximum(q - multipliers[1], 0)
        v3 = s + bias
        aux = q - multipliers[3]
        rhs = np.fft.fft2(to_fft_layout(aux))
        for axis, filter_ in enumerate(filters):
            rhs += filter_.conj() * np.fft.fft2(previous_gradient[axis] + old_dual[axis])
        frequency = rhs / denominator
        intermediate = np.fft.ifft2(frequency) * PIXELS
        gradient = np.stack([np.fft.ifft2(filter_ * frequency) for filter_ in filters])
        corrected = gradient.real - old_dual
        norm = np.sqrt(np.square(corrected[0]) + np.square(corrected[1]))
        scale = np.maximum(norm - TAU, 0) / (np.maximum(norm - TAU, 0) + TAU)
        new_gradient = corrected * scale
        new_dual = old_dual - (gradient.real - new_gradient)
        v4 = intermediate.real.transpose(0, 2, 1).reshape(CLASSES, PIXELS) / PIXELS
        if t == 2:
            save("q_kernel", [item for pair in zip(values, multipliers) for item in pair], [compact])
            save("fixq_kernel", [compact, targets], [q])
            save("nu_aux3_sum_kernel", [q, multipliers[2]], [s, sums])
            save("Vs_kernel", [posterior, q, *multipliers, s, bias], [v1, v2, v3, aux])
            # These two kernels in the paper process one spatial plane per launch.
            inner = intermediate[0].copy()
            inner.real /= PIXELS
            save("V4_kernel", [intermediate[0]], [inner, inner, inner, v4[0]])
            save("V5_kernel", [gradient[0, 0] * PIXELS, gradient[1, 0] * PIXELS,
                               old_dual[0, 0], old_dual[1, 0]],
                 [new_gradient[0, 0], new_gradient[1, 0], new_dual[0, 0], new_dual[1, 0]])
        updated = [v1, v2, v3, v4]
        multipliers = [d - (q - value) for d, value in zip(multipliers, updated)]
        values, previous_gradient, old_dual = updated, new_gradient, new_dual


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(image_path, label_path, out):
    image = loadmat(image_path)["paviaU"]
    labels = loadmat(label_path)["paviaU_gt"]
    if image.shape != (HEIGHT, WIDTH, 103) or labels.shape != (HEIGHT, WIDTH):
        raise ValueError("Unexpected Pavia University shape")

    def save(name, inputs, outputs):
        for arm, arrays in (("in", inputs), ("ref", outputs)):
            with (out / f"{name}.{arm}").open("xb") as stream:
                for array in arrays:
                    if not np.isfinite(array).all():
                        raise ValueError(f"{name} contains a nonfinite value")
                    np.asarray(array, dtype="<c16" if np.iscomplexobj(array) else "<f8").tofile(stream)

    posterior, chosen, targets = probabilities(image, labels)
    chosen.astype("<i4").tofile(out / "training_indices.i32")
    checkpoint(posterior, chosen, targets, save)
    plan = {
        "shape": [HEIGHT, WIDTH, 103], "classes": CLASSES, "training_samples": len(chosen),
        "seed": 20261124, "checkpoint_iteration": 3, "mu": MU, "tau": TAU, "target_class_index": 0,
        "source": "Wu et al. 10.1109/JSTARS.2017.2755639, reconstructed CUDA stages",
        "upstream": "RBF features with L1 multinomial logistic regression solved by CPU proximal gradient, scalar shrinkage threshold",
        "image_sha256": digest(image_path), "label_sha256": digest(label_path),
        "runtime_speedup_prediction": None, "hardware_miss_prediction": None,
        "pads": {"q_kernel": 0, "fixq_kernel": 0, "nu_aux3_sum_kernel": 0,
                 "Vs_kernel": 0, "V4_kernel": 0, "V5_kernel": 2},
        "pad_goal": "32 B sector alignment of contiguous warp reads; fixq irregular scatter has no proven nonzero candidate",
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    print(json.dumps(plan), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    prepare(args.image, args.labels, args.out)

