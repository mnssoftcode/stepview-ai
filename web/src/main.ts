/**
 * StepView Web — Main Application Entry Point
 *
 * Connects Camera feed to ONNX Runtime Web and the 2D Footstep Candidate Engine.
 * Optimized for mobile handheld use, portrait/landscape orientation, and controlled frame pacing.
 */

import { FootstepProposalEngineWeb } from "./footstepEngine.js";
import { WebRenderer } from "./renderer.js";
import { WebTerrainSegmenter } from "./segmenter.js";
import { FootstepCandidate, SegmentationResult } from "./types.js";

class StepViewWebApp {
  private videoEl: HTMLVideoElement;
  private canvasEl: HTMLCanvasElement;
  private segmenter: WebTerrainSegmenter;
  private footstepEngine: FootstepProposalEngineWeb;
  private renderer: WebRenderer;

  private isRunning: boolean = false;
  private isProcessing: boolean = false;
  private mediaStream: MediaStream | null = null;
  private lastProcessedSource: HTMLVideoElement | HTMLImageElement | null = null;

  // Frame pacing configuration
  private targetFps: number = 15; // 15 FPS default for thermal & battery stability on handheld mobile
  private lastProcessTime: number = 0;

  // UI elements
  private statusEl!: HTMLElement;
  private latencyEl!: HTMLElement;
  private fpsEl!: HTMLElement;
  private topStepEl!: HTMLElement;
  private providerEl!: HTMLElement;
  private bannerEl!: HTMLElement;
  private btnCamera!: HTMLButtonElement;
  private btnSampleTrail!: HTMLButtonElement;
  private btnSamplePark!: HTMLButtonElement;
  private btnSampleVillage!: HTMLButtonElement;
  private fileInput!: HTMLInputElement;
  private chkSeg!: HTMLInputElement;
  private chkProposals!: HTMLInputElement;
  private sliderAlpha!: HTMLInputElement;
  private selectFps!: HTMLSelectElement;

  // Real-time FPS measurement
  private fpsCounter: number = 0;
  private currentFps: number = 0;
  private lastFpsUpdate: number = performance.now();

  constructor() {
    this.videoEl = document.getElementById("video-feed") as HTMLVideoElement;
    this.canvasEl = document.getElementById("main-canvas") as HTMLCanvasElement;

    this.segmenter = new WebTerrainSegmenter();
    this.footstepEngine = new FootstepProposalEngineWeb({
      lookaheadMinYRatio: 0.35,
      lookaheadMaxYRatio: 0.92,
      nearFootprintSize: [54, 32],
      farFootprintSize: [26, 16],
      minClearancePx: 10.0,
      idealClearancePx: 40.0,
      minGroundSupportRatio: 0.70,
      topK: 5,
    });
    this.renderer = new WebRenderer(this.canvasEl);

    this.initUI();
  }

  private initUI(): void {
    this.statusEl = document.getElementById("status-text")!;
    this.latencyEl = document.getElementById("metric-latency")!;
    this.fpsEl = document.getElementById("metric-fps")!;
    this.topStepEl = document.getElementById("metric-top-step")!;
    this.providerEl = document.getElementById("metric-provider")!;
    this.bannerEl = document.getElementById("alert-banner")!;

    this.btnCamera = document.getElementById("btn-toggle-camera") as HTMLButtonElement;
    this.btnSampleTrail = document.getElementById("btn-sample-trail") as HTMLButtonElement;
    this.btnSamplePark = document.getElementById("btn-sample-park") as HTMLButtonElement;
    this.btnSampleVillage = document.getElementById("btn-sample-village") as HTMLButtonElement;
    this.fileInput = document.getElementById("file-upload") as HTMLInputElement;

    this.chkSeg = document.getElementById("chk-show-seg") as HTMLInputElement;
    this.chkProposals = document.getElementById("chk-show-proposals") as HTMLInputElement;
    this.sliderAlpha = document.getElementById("slider-alpha") as HTMLInputElement;
    this.selectFps = document.getElementById("select-fps") as HTMLSelectElement;

    this.btnCamera.addEventListener("click", () => this.toggleCamera());
    this.btnSampleTrail.addEventListener("click", () => this.loadSampleImage("/samples/trail.png"));
    this.btnSamplePark.addEventListener("click", () => this.loadSampleImage("/samples/park.png"));
    this.btnSampleVillage.addEventListener("click", () => this.loadSampleImage("/samples/village.png"));

    this.fileInput.addEventListener("change", (e) => this.handleFileUpload(e));

    this.chkSeg.addEventListener("change", () => this.rerenderCurrentSource());
    this.chkProposals.addEventListener("change", () => this.rerenderCurrentSource());
    this.sliderAlpha.addEventListener("input", () => this.rerenderCurrentSource());

    this.selectFps.addEventListener("change", () => {
      this.targetFps = parseInt(this.selectFps.value, 10) || 15;
    });

    // Check secure context for camera
    this.checkSecureContext();
  }

  private checkSecureContext(): void {
    const isLocalhost =
      window.location.hostname === "localhost" ||
      window.location.hostname === "127.0.0.1" ||
      window.location.hostname === "::1";
    const isHttps = window.location.protocol === "https:";

    if (!isHttps && !isLocalhost) {
      this.showAlert(
        "Mobile Notice: Camera access requires HTTPS on mobile devices. If accessing on a phone via local IP, use test images or deploy to Render (HTTPS)."
      );
    }
  }

  private showAlert(msg: string): void {
    if (this.bannerEl) {
      this.bannerEl.textContent = msg;
      this.bannerEl.style.display = "block";
    }
  }

  private hideAlert(): void {
    if (this.bannerEl) {
      this.bannerEl.style.display = "none";
    }
  }

  public async start(): Promise<void> {
    this.statusEl.textContent = "Loading ONNX model (stepview_segmentation.onnx)...";

    try {
      await this.segmenter.loadModel("/models/stepview_segmentation.onnx");
      this.statusEl.textContent = "Model ready. Interactive.";
      this.providerEl.textContent = this.segmenter.activeProvider.toUpperCase();
      this.btnCamera.disabled = false;
      this.btnSampleTrail.disabled = false;
      this.btnSamplePark.disabled = false;
      this.btnSampleVillage.disabled = false;

      // Automatically load default trail image for immediate visualization
      await this.loadSampleImage("/samples/trail.png");
    } catch (err) {
      console.error("[StepView] Model initialization failed:", err);
      const rawMsg = err instanceof Error ? err.message : String(err);
      if (rawMsg.startsWith("MODEL LOAD ERROR")) {
        this.statusEl.textContent = `[MODEL LOAD ERROR] ${rawMsg.replace("MODEL LOAD ERROR: ", "")}`;
      } else if (rawMsg.startsWith("INFERENCE BACKEND ERROR")) {
        this.statusEl.textContent = `[BACKEND ERROR] ${rawMsg.replace("INFERENCE BACKEND ERROR: ", "")}`;
      } else {
        this.statusEl.textContent = `Error: ${rawMsg}`;
      }
      this.showAlert(rawMsg);
    }
  }

  private async toggleCamera(): Promise<void> {
    if (this.isRunning) {
      this.stopCamera();
    } else {
      await this.startCamera();
    }
  }

  private async startCamera(): Promise<void> {
    this.statusEl.textContent = "Requesting rear camera access...";

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const msg = "CAMERA ERROR: Camera API not supported in this context (HTTPS required on mobile). Use Test Samples or File Upload below.";
      this.statusEl.textContent = "[CAMERA ERROR] MediaDevices API unsupported.";
      this.showAlert(msg);
      return;
    }

    try {
      // Request environment (back) camera with standard mobile aspect ratio
      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.mediaStream = stream;
      this.videoEl.srcObject = stream;
      await this.videoEl.play();

      this.isRunning = true;
      this.btnCamera.textContent = "Stop Camera";
      this.btnCamera.classList.add("active");
      this.statusEl.textContent = "Live camera stream active.";
      this.hideAlert();

      // Adjust canvas to match native camera orientation (portrait or landscape)
      this.updateCanvasDimensions(this.videoEl.videoWidth || 640, this.videoEl.videoHeight || 480);

      this.videoEl.onloadedmetadata = () => {
        this.updateCanvasDimensions(this.videoEl.videoWidth, this.videoEl.videoHeight);
      };

      this.lastProcessTime = 0;
      this.scheduleLoop();
    } catch (err) {
      console.warn("Camera access failed:", err);
      const msg = err instanceof Error ? err.message : String(err);
      this.statusEl.textContent = `[CAMERA ERROR] ${msg}`;
      this.showAlert(`CAMERA ERROR: Camera access unavailable: ${msg}. Try the sample terrain images or image upload.`);
    }
  }

  private stopCamera(): void {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }
    this.isRunning = false;
    this.btnCamera.textContent = "Start Camera";
    this.btnCamera.classList.remove("active");
    this.statusEl.textContent = "Camera stopped.";
  }

  private updateCanvasDimensions(sourceW: number, sourceH: number): void {
    this.canvasEl.width = sourceW;
    this.canvasEl.height = sourceH;
  }

  private async loadSampleImage(url: string): Promise<void> {
    this.stopCamera();
    this.statusEl.textContent = `Loading test image (${url.split("/").pop()})...`;

    const img = new Image();
    img.src = url;
    await img.decode();

    this.updateCanvasDimensions(img.naturalWidth || 640, img.naturalHeight || 480);
    this.lastProcessedSource = img;
    await this.processSingleFrame(img);
    this.statusEl.textContent = `Active Sample: ${url.split("/").pop()}`;
  }

  private handleFileUpload(e: Event): void {
    const input = e.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;

    const file = input.files[0];
    const reader = new FileReader();

    reader.onload = async (event) => {
      const dataUrl = event.target?.result as string;
      const img = new Image();
      img.onload = async () => {
        this.stopCamera();
        this.updateCanvasDimensions(img.naturalWidth, img.naturalHeight);
        this.lastProcessedSource = img;
        await this.processSingleFrame(img);
        this.statusEl.textContent = `Uploaded: ${file.name}`;
      };
      img.src = dataUrl;
    };

    reader.readAsDataURL(file);
  }

  private async rerenderCurrentSource(): Promise<void> {
    if (!this.isRunning && this.lastProcessedSource) {
      await this.processSingleFrame(this.lastProcessedSource);
    }
  }

  private scheduleLoop(): void {
    if (!this.isRunning) return;

    requestAnimationFrame(async () => {
      if (this.isRunning) {
        const now = performance.now();
        const minInterval = 1000.0 / this.targetFps;

        // Controlled pacing: only trigger inference when minInterval elapsed and worker idle
        if (!this.isProcessing && now - this.lastProcessTime >= minInterval) {
          this.lastProcessTime = now;
          this.lastProcessedSource = this.videoEl;
          await this.processSingleFrame(this.videoEl);
        }
        this.scheduleLoop();
      }
    });
  }

  private async processSingleFrame(
    source: HTMLVideoElement | HTMLImageElement
  ): Promise<void> {
    if (!this.segmenter.isLoaded() || this.isProcessing) return;

    this.isProcessing = true;
    const t0 = performance.now();

    try {
      // 1. ONNX Semantic Segmentation (256x256)
      const segResult: SegmentationResult = await this.segmenter.predict(source);

      // 2. 2D Footstep Candidate Engine
      const proposalResult = this.footstepEngine.proposeFootsteps(
        segResult.classMask,
        segResult.confidenceMap,
        segResult.width,
        segResult.height
      );

      const totalPipelineMs = performance.now() - t0;
      const alpha = parseFloat(this.sliderAlpha.value) || 0.38;

      // 3. Canvas Overlay Rendering
      this.renderer.render(
        source,
        segResult,
        proposalResult.topCandidates,
        {
          showSegmentation: this.chkSeg.checked,
          segmentationAlpha: alpha,
          showCandidates: this.chkProposals.checked,
          showLegend: true,
        }
      );

      // 4. Update Dashboard Metrics
      this.updateMetrics(totalPipelineMs, segResult.latencyMs, proposalResult.topCandidates);
    } catch (err) {
      console.error("Frame processing failed:", err);
    } finally {
      this.isProcessing = false;
    }
  }

  private updateMetrics(
    totalPipelineMs: number,
    onnxLatencyMs: number,
    candidates: FootstepCandidate[]
  ): void {
    this.latencyEl.textContent = `${totalPipelineMs.toFixed(0)} ms (ONNX: ${onnxLatencyMs.toFixed(0)} ms)`;

    // FPS calculation
    this.fpsCounter++;
    const now = performance.now();
    if (now - this.lastFpsUpdate >= 1000) {
      this.currentFps = Math.round((this.fpsCounter * 1000) / (now - this.lastFpsUpdate));
      this.fpsCounter = 0;
      this.lastFpsUpdate = now;
      this.fpsEl.textContent = `${this.currentFps} FPS`;
    }

    // Top candidate details
    if (candidates.length > 0) {
      const top = candidates[0];
      this.topStepEl.textContent = `#1 (Score: ${top.score.toFixed(2)}, Clr: ${top.obstacle_clearance_px.toFixed(0)}px)`;
      this.topStepEl.style.color = "#ffd700";
    } else {
      this.topStepEl.textContent = "Refusal (Hazard detected)";
      this.topStepEl.style.color = "#f85149";
    }
  }
}

// Bootstrap on DOM ready
window.addEventListener("DOMContentLoaded", () => {
  const app = new StepViewWebApp();
  app.start().catch((err) => console.error("Initialization error:", err));
});
