/**
 * StepView Web — ONNX Runtime Web Segmentation Engine
 *
 * Runs mobile-optimized MobileNetV3-Small LR-ASPP model directly in the browser
 * using ONNX Runtime Web with WebGPU / WebAssembly acceleration.
 */

import * as ort from "onnxruntime-web";
import { SegmentationResult } from "./types.js";

// Point wasm assets to the web root
ort.env.wasm.wasmPaths = "/";
ort.env.wasm.numThreads = 1; // Single-thread mode guarantees universal browser compatibility

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

  public async loadModel(modelUrl: string = "/models/stepview_segmentation.onnx"): Promise<void> {
    console.log(`[StepView] Loading ONNX model from: ${modelUrl}`);

    // Try WebGPU first if supported, fallback to Wasm
    const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;
    const preferredProviders = hasWebGPU ? ["webgpu", "wasm"] : ["wasm"];

    try {
      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: preferredProviders,
        graphOptimizationLevel: "all",
      });
      this.activeProvider = hasWebGPU ? "webgpu" : "wasm";
      console.log(`[StepView] ONNX model successfully loaded with provider: ${this.activeProvider}`);
    } catch (err) {
      console.warn("[StepView] Preferred provider failed, falling back to pure WASM:", err);
      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ["wasm"],
      });
      this.activeProvider = "wasm";
      console.log("[StepView] ONNX model successfully loaded with fallback WASM provider.");
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
