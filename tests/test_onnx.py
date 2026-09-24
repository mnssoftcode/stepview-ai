"""Automated tests for StepView ONNX model loading, inference, output shapes, and valid class IDs."""

from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from stepview.data.schema import NUM_CLASSES, VALID_CLASS_IDS
from stepview.inference.onnx_segmenter import ONNXTerrainSegmenter

ONNX_MODEL_PATH = Path("models/stepview_segmentation.onnx")


@pytest.fixture(scope="module")
def onnx_model_path() -> Path:
    """Fixture to ensure the exported ONNX model exists."""
    assert ONNX_MODEL_PATH.is_file(), (
        f"Exported ONNX model not found at {ONNX_MODEL_PATH}. "
        "Run model export before running ONNX tests."
    )
    return ONNX_MODEL_PATH


def test_onnx_file_validity(onnx_model_path: Path) -> None:
    """Verify that the ONNX model file loads cleanly and passes onnx.checker validation."""
    model = onnx.load(str(onnx_model_path))
    onnx.checker.check_model(model)
    assert model.ir_version > 0
    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 1


def test_onnx_model_loading(onnx_model_path: Path) -> None:
    """Verify that ONNX Runtime session loads model with CPUExecutionProvider."""
    session = ort.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    assert len(inputs) == 1
    assert len(outputs) == 1

    input_meta = inputs[0]
    output_meta = outputs[0]

    assert input_meta.name == "input"
    assert input_meta.type == "tensor(float)"
    # Shape: (batch_size, 3, 256, 256)
    assert len(input_meta.shape) == 4
    assert input_meta.shape[1:] == [3, 256, 256]

    assert output_meta.name in ("logits", "output")
    assert output_meta.type == "tensor(float)"
    # Shape: 4D tensor (batch_size, C, H, W)
    assert len(output_meta.shape) == 4


def test_onnx_inference_and_output_shape(onnx_model_path: Path) -> None:
    """Verify inference execution and output shape on single and batch inputs."""
    session = ort.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name

    # Single-sample inference (1, 3, 256, 256)
    dummy_single = np.random.uniform(0.0, 1.0, (1, 3, 256, 256)).astype(np.float32)
    output_single = session.run([output_name], {input_name: dummy_single})[0]

    assert isinstance(output_single, np.ndarray)
    assert output_single.shape == (1, NUM_CLASSES, 256, 256)
    assert output_single.dtype == np.float32
    assert np.all(np.isfinite(output_single)), "Outputs must contain finite numbers without NaN/Inf"

    # Multi-sample batch inference (2, 3, 256, 256)
    dummy_batch = np.random.uniform(0.0, 1.0, (2, 3, 256, 256)).astype(np.float32)
    output_batch = session.run([output_name], {input_name: dummy_batch})[0]

    assert output_batch.shape == (2, NUM_CLASSES, 256, 256)
    assert np.all(np.isfinite(output_batch))


def test_onnx_valid_class_ids_and_probabilities(onnx_model_path: Path) -> None:
    """Verify that predictions map to valid class IDs and valid probability distributions."""
    session = ort.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.uniform(0.0, 1.0, (1, 3, 256, 256)).astype(np.float32)
    raw_logits = session.run([output_name], {input_name: dummy_input})[0]  # (1, 6, 256, 256)

    # Argmax class predictions
    pred_classes = np.argmax(raw_logits, axis=1)  # (1, 256, 256)
    unique_ids = set(np.unique(pred_classes).tolist())

    # All predicted IDs must be a subset of valid class IDs {0, 1, 2, 3, 4, 5}
    assert unique_ids.issubset(set(VALID_CLASS_IDS))

    # Softmax probabilities
    exp_logits = np.exp(raw_logits - np.max(raw_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Probabilities must be within [0, 1] and sum to 1.0 along channel axis
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    prob_sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5, atol=1e-5)


def test_onnx_terrain_segmenter_wrapper(onnx_model_path: Path) -> None:
    """Verify high-level ONNXTerrainSegmenter end-to-end wrapper on arbitrary image sizes."""
    segmenter = ONNXTerrainSegmenter(onnx_model_path)

    # Test with arbitrary test image dimensions (e.g. 120 x 180 RGB)
    test_img = np.random.randint(0, 256, (120, 180, 3), dtype=np.uint8)
    result = segmenter.predict(test_img)

    assert "class_mask" in result
    assert "confidence_map" in result
    assert "probabilities" in result
    assert "latency_ms" in result
    assert "raw_output_shape" in result

    class_mask = result["class_mask"]
    conf_map = result["confidence_map"]
    probs = result["probabilities"]

    assert class_mask.shape == (120, 180)
    assert conf_map.shape == (120, 180)
    assert probs.shape == (NUM_CLASSES, 120, 180)

    # Output class IDs must be valid
    unique_ids = set(np.unique(class_mask).tolist())
    assert unique_ids.issubset(set(VALID_CLASS_IDS))

    # Confidence must be bounded [0, 1]
    assert np.all(conf_map >= 0.0)
    assert np.all(conf_map <= 1.0)
    assert result["latency_ms"] > 0.0


def test_onnx_footstep_proposals_integration(onnx_model_path: Path) -> None:
    """Verify that ONNX segmentation outputs pipe directly into FootstepProposalEngine."""
    from stepview.geometry.footstep_proposals import FootstepProposalEngine

    segmenter = ONNXTerrainSegmenter(onnx_model_path)
    engine = FootstepProposalEngine(top_k=3)

    # Use a real terrain image if available, else synthetic
    sample_img_path = Path("data/images/rugd_trail-4_frame00586.png")
    if sample_img_path.is_file():
        import cv2
        bgr = cv2.imread(str(sample_img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = np.full((256, 256, 3), 120, dtype=np.uint8)

    h, w = rgb.shape[:2]
    seg_res = segmenter.predict(rgb)

    proposals = engine.propose_footsteps(
        class_mask=seg_res["class_mask"],
        confidence_map=seg_res["confidence_map"],
        probabilities=seg_res["probabilities"],
    )

    assert "top_candidates" in proposals
    assert "usable_ground_mask" in proposals
    assert "hazard_distance_map" in proposals
    assert "summary" in proposals

    assert proposals["usable_ground_mask"].shape == (h, w)
    assert proposals["hazard_distance_map"].shape == (h, w)
    assert len(proposals["top_candidates"]) <= 3

    for cand in proposals["top_candidates"]:
        assert 0.0 <= cand.x_center <= w
        assert 0.0 <= cand.y_center <= h
        assert 0.0 <= cand.score <= 1.0
        assert cand.is_valid

