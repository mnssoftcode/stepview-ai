/**
 * StepView Web — ONNX Runtime Web Segmentation Engine
 *
 * Runs mobile-optimized MobileNetV3-Small LR-ASPP model directly in the browser
 * using ONNX Runtime Web with WebGPU / WebAssembly acceleration.
 */

import * as ort from "onnxruntime-web";
import { SegmentationResult } from "./types.js";

// Configure ONNX Runtime Web asset paths
if (typeof window !== "undefined") {
  const origin = window.location.origin ? window.location.origin + "/" : "/";
  ort.env.wasm.wasmPaths = origin;
  // Default to single-thread mode for universal compatibility unless cross-origin isolation is active
  ort.env.wasm.numThreads = window.crossOriginIsolated && typeof SharedArrayBuffer !== "undefined"
    ? Math.min(4, navigator.hardwareConcurrency || 2)
    : 1;
}

export class WebTerrainSegmenter {
  private session: ort.InferenceSession | null = null;
  private offscreenCanvas: HTMLCanvasElement;
  private offscreenCtx: CanvasRenderingContext2D;
  public readonly targetSize = 256;
  public activeProvider: string = "wasm";

  constructor() {
    this.offscreenCanvas = document.createElement("canvas");
    this.offscreenCanvas.width = this.targetSize;
    this.offscreenCanvas.height = this.targetSize;
    const ctx = this.offscreenCanvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      throw new Error("Failed to initialize 2D canvas context");
    }
    this.offscreenCtx = ctx;
  }

  public isLoaded(): boolean {
    return this.session !== null;
  }

  /**
   * Helper to check if an asset URL is accessible on the server.
   */
  private async checkAssetExists(url: string): Promise<boolean> {
    try {
      const res = await fetch(url, { method: "HEAD" });
      return res.ok;
    } catch {
      return false;
    }
  }

  public async loadModel(modelUrl: string = "/models/stepview_segmentation.onnx"): Promise<void> {
    console.log(`[StepView] Initializing ONNX Runtime Web for model: ${modelUrl}`);

    // 1. Separate Model Load verification: fetch model bytes first.
    // This cleanly separates MODEL LOAD ERROR from INFERENCE BACKEND ERROR.
    let modelBuffer: ArrayBuffer;
    try {
      const resp = await fetch(modelUrl);
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status} (${resp.statusText || "Resource Unavailable"})`);
      }
      modelBuffer = await resp.arrayBuffer();
      if (!modelBuffer || modelBuffer.byteLength === 0) {
        throw new Error("Model file is empty (0 bytes)");
      }
      console.log(
        `[StepView] Model binary downloaded successfully (${(modelBuffer.byteLength / 1024 / 1024).toFixed(2)} MB)`
      );
    } catch (fetchErr) {
      const msg = fetchErr instanceof Error ? fetchErr.message : String(fetchErr);
      throw new Error(`MODEL LOAD ERROR: Model file could not be loaded: ${msg}`);
    }

    const basePath = typeof window !== "undefined" && window.location?.origin
      ? window.location.origin + "/"
      : "/";
    ort.env.wasm.wasmPaths = basePath;

    // 2. Check cross-origin isolation and configure threading safely
    const isCrossIsolated = typeof window !== "undefined" && Boolean(window.crossOriginIsolated);
    const hasSharedArrayBuffer = typeof SharedArrayBuffer !== "undefined";
    const canUseThreads = isCrossIsolated && hasSharedArrayBuffer;

    if (canUseThreads) {
      const threads = Math.min(4, typeof navigator !== "undefined" ? (navigator.hardwareConcurrency || 2) : 2);
      ort.env.wasm.numThreads = threads;
      console.log(`[StepView] Cross-origin isolated: enabled ${threads} WASM worker threads.`);
    } else {
      ort.env.wasm.numThreads = 1;
      console.log("[StepView] Universal single-threaded WASM mode active (cross-origin isolation inactive).");
    }

    // 3. Pre-flight check on runtime assets to ensure availability
    const wasmReady = await this.checkAssetExists(`${basePath}ort-wasm-simd-threaded.wasm`);
    const mjsReady = await this.checkAssetExists(`${basePath}ort-wasm-simd-threaded.mjs`);
    if (!wasmReady && !mjsReady) {
      console.warn(`[StepView] Pre-flight warning: WASM assets not verified at ${basePath}`);
    }

    // 4. Tiered Provider Selection:
    // Preferred: WebGPU -> WASM SIMD/threaded (when available) -> standard WASM fallback
    const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;
    let jsepReady = false;
    if (hasWebGPU) {
      jsepReady = await this.checkAssetExists(`${basePath}ort-wasm-simd-threaded.jsep.mjs`);
    }

    const modelBytes = new Uint8Array(modelBuffer);

    // Tier 1: WebGPU
    if (hasWebGPU && jsepReady) {
      try {
        console.log("[StepView] Attempting WebGPU provider...");
        this.session = await ort.InferenceSession.create(modelBytes, {
          executionProviders: ["webgpu"],
          graphOptimizationLevel: "all",
        });
        this.activeProvider = "webgpu";
        console.log("[StepView] ONNX model successfully initialized with provider: webgpu");
        return;
      } catch (gpuErr) {
        console.warn("[StepView] WebGPU provider failed, falling back to WASM:", gpuErr);
      }
    }

    // Tier 2: WASM (SIMD / Threaded if supported)
    if (canUseThreads) {
      try {
        console.log("[StepView] Attempting multi-threaded WASM provider...");
        this.session = await ort.InferenceSession.create(modelBytes, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
        this.activeProvider = "wasm-simd";
        console.log("[StepView] ONNX model successfully initialized with provider: wasm-simd");
        return;
      } catch (simdErr) {
        console.warn("[StepView] Multi-threaded WASM failed, falling back to single-threaded WASM:", simdErr);
      }
    }

    // Tier 3: Universal Single-Threaded WASM Fallback
    try {
      console.log("[StepView] Attempting single-threaded WASM fallback...");
      ort.env.wasm.numThreads = 1;
      this.session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      this.activeProvider = "wasm";
      console.log("[StepView] ONNX model successfully initialized with fallback provider: wasm");
      return;
    } catch (wasmErr) {
      console.error("[StepView] WASM backend initialization failed:", wasmErr);
      const msg = wasmErr instanceof Error ? wasmErr.message : String(wasmErr);
      throw new Error(`INFERENCE BACKEND ERROR: ONNX Runtime WASM failed to initialize: ${msg}`);
    }
  }

  /**
   * Run semantic segmentation on an HTMLVideoElement, HTMLImageElement, or Canvas.
   */
  public async predict(
    source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement
  ): Promise<SegmentationResult> {
    if (!this.session) {
      throw new Error("Model is not loaded. Call loadModel() first.");
    }

    // 1. Draw source into 256x256 offscreen canvas
    this.offscreenCtx.drawImage(source, 0, 0, this.targetSize, this.targetSize);
    const imgData = this.offscreenCtx.getImageData(0, 0, this.targetSize, this.targetSize);
    const rgba = imgData.data;

    // 2. Preprocess: convert HWC RGBA to CHW float32 normalized [0.0, 1.0]
    const pixels = this.targetSize * this.targetSize;
    const floatArr = new Float32Array(3 * pixels);

    for (let i = 0; i < pixels; i++) {
      floatArr[i] = rgba[i * 4] / 255.0; // Red
      floatArr[pixels + i] = rgba[i * 4 + 1] / 255.0; // Green
      floatArr[2 * pixels + i] = rgba[i * 4 + 2] / 255.0; // Blue
    }

    // 3. Create ONNX Tensor
    const inputTensor = new ort.Tensor("float32", floatArr, [1, 3, this.targetSize, this.targetSize]);

    // 4. Execute inference
    const tInfer0 = performance.now();
    const feeds: Record<string, ort.Tensor> = { input: inputTensor };
    const results = await this.session.run(feeds);
    const inferLatencyMs = performance.now() - tInfer0;

    // 5. Output tensor logits shape: [1, 6, 256, 256]
    const outputTensor = results.logits || results.output || Object.values(results)[0];
    const logits = outputTensor.data as Float32Array;

    // 6. Compute softmax probabilities, argmax class mask, and confidence map
    const classMask = new Uint8Array(pixels);
    const confidenceMap = new Float32Array(pixels);
    const numClasses = 6;

    for (let i = 0; i < pixels; i++) {
      // Find max logit for numerical stability
      let maxLogit = -Infinity;
      for (let c = 0; c < numClasses; c++) {
        const val = logits[c * pixels + i];
        if (val > maxLogit) maxLogit = val;
      }

      // Exp and sum
      let sumExp = 0.0;
      let bestClass = 0;
      let bestProb = -Infinity;

      for (let c = 0; c < numClasses; c++) {
        const prob = Math.exp(logits[c * pixels + i] - maxLogit);
        sumExp += prob;
      }

      for (let c = 0; c < numClasses; c++) {
        const prob = Math.exp(logits[c * pixels + i] - maxLogit) / sumExp;
        if (prob > bestProb) {
          bestProb = prob;
          bestClass = c;
        }
      }

      classMask[i] = bestClass;
      confidenceMap[i] = bestProb;
    }

    return {
      classMask,
      confidenceMap,
      probabilities: logits,
      width: this.targetSize,
      height: this.targetSize,
      latencyMs: inferLatencyMs,
    };
  }
}
