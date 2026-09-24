"""
Tests for StepView Web production asset integrity and ONNX Runtime Web configuration.

Validates:
1. ONNX model availability and matching paths.
2. WebAssembly (.wasm) and ES module loader (.mjs) assets in web/public and web/dist.
3. Production build configuration in render.yaml and package.json.
4. Robust ONNX Runtime Web initialization configuration in segmenter.ts.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
PUBLIC_DIR = WEB_DIR / "public"
DIST_DIR = WEB_DIR / "dist"


def test_onnx_model_availability():
    """Verify primary and web-public ONNX models exist and are valid sizes."""
    primary_model = REPO_ROOT / "models" / "stepview_segmentation.onnx"
    public_model = PUBLIC_DIR / "models" / "stepview_segmentation.onnx"

    assert primary_model.exists(), f"Primary ONNX model missing: {primary_model}"
    assert primary_model.stat().st_size > 1_000_000, "Primary ONNX model is too small (<1MB)"

    assert public_model.exists(), f"Web public ONNX model missing: {public_model}"
    assert public_model.stat().st_size == primary_model.stat().st_size, (
        f"Public model size ({public_model.stat().st_size}) does not match primary model ({primary_model.stat().st_size})"
    )


def test_wasm_and_mjs_asset_availability():
    """Verify required ONNX Runtime Web wasm and mjs companion assets exist in public and dist."""
    required_assets = [
        "ort-wasm-simd-threaded.wasm",
        "ort-wasm-simd-threaded.mjs",
        "ort-wasm-simd-threaded.jsep.wasm",
        "ort-wasm-simd-threaded.jsep.mjs",
    ]

    for asset in required_assets:
        pub_file = PUBLIC_DIR / asset
        assert pub_file.exists(), f"Required runtime asset missing in public/: {asset}"
        assert pub_file.stat().st_size > 0, f"Runtime asset is empty in public/: {asset}"

        dist_file = DIST_DIR / asset
        assert dist_file.exists(), f"Required runtime asset missing in dist/: {asset}"
        assert dist_file.stat().st_size == pub_file.stat().st_size, (
            f"Asset in dist/ does not match public/: {asset}"
        )


def test_copy_wasm_script_handles_both_wasm_and_mjs():
    """Verify copy-wasm.js copies both .wasm and .mjs files."""
    script_path = WEB_DIR / "scripts" / "copy-wasm.js"
    assert script_path.exists()
    content = script_path.read_text()
    assert ".wasm" in content
    assert ".mjs" in content
    assert "ort-wasm" in content


def test_render_yaml_configuration():
    """Verify render.yaml specifies correct build command and staticPublishPath."""
    render_yaml_path = REPO_ROOT / "render.yaml"
    assert render_yaml_path.exists()
    content = render_yaml_path.read_text()

    assert "type: web" in content
    assert "env: static" in content
    assert "npm run build" in content
    assert "staticPublishPath: ./web/dist" in content
    assert "Cross-Origin-Opener-Policy" in content
    assert "Cross-Origin-Embedder-Policy" in content



def test_runtime_initialization_configuration():
    """Verify segmenter.ts explicitly configures asset paths, single-thread fallback, and distinct error reporting."""
    segmenter_path = WEB_DIR / "src" / "segmenter.ts"
    assert segmenter_path.exists()
    code = segmenter_path.read_text()

    # Asset paths configuration
    assert "ort.env.wasm.wasmPaths" in code
    # Safe threading configuration
    assert "ort.env.wasm.numThreads" in code
    assert "window.crossOriginIsolated" in code
    # Tiered execution providers
    assert "webgpu" in code
    assert "wasm" in code
    # Model load vs backend error distinction
    assert "MODEL LOAD ERROR" in code
    assert "INFERENCE BACKEND ERROR" in code
