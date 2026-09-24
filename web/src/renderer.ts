/**
 * StepView Web — Canvas Overlay & Footstep Visualizer
 *
 * Renders live video feeds, semantic terrain segmentation masks, and
 * ranked footstep placement candidates directly on HTML5 Canvas.
 */

import { CLASS_PALETTE, FootstepCandidate, SegmentationResult, TerrainClass } from "./types.js";

export interface RenderOptions {
  showSegmentation: boolean;
  segmentationAlpha: number;
  showCandidates: boolean;
  showLegend: boolean;
}

function safeRoundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
): void {
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(x, y, w, h, r);
  } else {
    // Fallback for older browsers
    ctx.rect(x, y, w, h);
  }
}

export class WebRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private maskCanvas: HTMLCanvasElement;
  private maskCtx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Failed to get 2D context");
    this.ctx = ctx;

    this.maskCanvas = document.createElement("canvas");
    this.maskCanvas.width = 256;
    this.maskCanvas.height = 256;
    const mCtx = this.maskCanvas.getContext("2d");
    if (!mCtx) throw new Error("Failed to get mask 2D context");
    this.maskCtx = mCtx;
  }

  public render(
    source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    segResult: SegmentationResult | null,
    candidates: FootstepCandidate[],
    options: RenderOptions
  ): void {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctx = this.ctx;

    // 1. Draw base video / image frame
    ctx.drawImage(source, 0, 0, w, h);

    // 2. Draw segmentation mask overlay if available and enabled
    if (segResult && options.showSegmentation) {
      this.drawSegmentationOverlay(segResult, options.segmentationAlpha);
      ctx.drawImage(this.maskCanvas, 0, 0, w, h);
    }

    // 3. Draw footstep candidates
    if (options.showCandidates && candidates.length > 0) {
      this.drawFootstepCandidates(candidates, w, h);
    } else if (segResult && candidates.length === 0) {
      this.drawRefusalBanner(w, h);
    }

    // 4. Draw legend banner
    if (options.showLegend) {
      this.drawLegend(w, h);
    }
  }

  private drawSegmentationOverlay(seg: SegmentationResult, alpha: number): void {
    const imgData = this.maskCtx.createImageData(seg.width, seg.height);
    const data = imgData.data;
    const mask = seg.classMask;
    const total = seg.width * seg.height;
    const alphaInt = Math.floor(alpha * 255);

    for (let i = 0; i < total; i++) {
      const cls = mask[i] as TerrainClass;
      const rgb = CLASS_PALETTE[cls] || [0, 0, 0];
      const p = i * 4;

      if (cls === TerrainClass.BACKGROUND) {
        // Keep background transparent
        data[p] = 0;
        data[p + 1] = 0;
        data[p + 2] = 0;
        data[p + 3] = 0;
      } else {
        data[p] = rgb[0];
        data[p + 1] = rgb[1];
        data[p + 2] = rgb[2];
        data[p + 3] = alphaInt;
      }
    }

    this.maskCtx.putImageData(imgData, 0, 0);
  }

  private drawFootstepCandidates(candidates: FootstepCandidate[], canvasW: number, canvasH: number): void {
    const ctx = this.ctx;
    const scaleX = canvasW / 256.0;
    const scaleY = canvasH / 256.0;

    // Draw in reverse rank order so Rank #1 is drawn on top
    for (let i = candidates.length - 1; i >= 0; i--) {
      const cand = candidates[i];
      const isTop = cand.rank === 1;

      const cx = cand.x_center * scaleX;
      const cy = cand.y_center * scaleY;
      const rx = (cand.width / 2.0) * scaleX;
      const ry = (cand.height / 2.0) * scaleY;

      ctx.save();
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);

      if (isTop) {
        // Gold / Amber for #1
        ctx.fillStyle = "rgba(255, 215, 0, 0.42)";
        ctx.fill();
        ctx.lineWidth = 3.5;
        ctx.strokeStyle = "#ffd700";
        ctx.stroke();

        // Inner white accent
        ctx.beginPath();
        ctx.ellipse(cx, cy, Math.max(1, rx - 3), Math.max(1, ry - 3), 0, 0, 2 * Math.PI);
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.90)";
        ctx.stroke();

        // Center crosshair
        ctx.beginPath();
        ctx.strokeStyle = "#ff2222";
        ctx.lineWidth = 2;
        ctx.moveTo(cx - 8, cy);
        ctx.lineTo(cx + 8, cy);
        ctx.moveTo(cx, cy - 8);
        ctx.lineTo(cx, cy + 8);
        ctx.stroke();

        // Top Step Badge
        const badge = `TOP STEP #1 (${cand.score.toFixed(2)})`;
        ctx.font = "bold 13px system-ui, -apple-system, sans-serif";
        const metrics = ctx.measureText(badge);
        const bw = metrics.width + 16;
        const bh = 24;
        let bx = cx - bw / 2;
        let by = cy - ry - bh - 8;
        if (by < 10) by = cy + ry + 8;

        ctx.fillStyle = "rgba(15, 15, 15, 0.90)";
        ctx.beginPath();
        safeRoundRect(ctx, bx, by, bw, bh, 5);
        ctx.fill();
        ctx.strokeStyle = "#ffd700";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.fillStyle = "#ffd700";
        ctx.fillText(badge, bx + 8, by + 17);
      } else {
        // Soft Emerald for alternatives
        ctx.fillStyle = "rgba(46, 204, 113, 0.25)";
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#2ecc71";
        ctx.stroke();

        // Compact rank badge
        const badge = `#${cand.rank} (${cand.score.toFixed(2)})`;
        ctx.font = "11px system-ui, -apple-system, sans-serif";
        const metrics = ctx.measureText(badge);
        const bw = metrics.width + 10;
        const bh = 18;
        const bx = cx - rx;
        const by = cy - ry - bh - 2;

        ctx.fillStyle = "rgba(18, 18, 18, 0.85)";
        ctx.beginPath();
        safeRoundRect(ctx, bx, by, bw, bh, 4);
        ctx.fill();
        ctx.fillStyle = "#ffffff";
        ctx.fillText(badge, bx + 5, by + 13);
      }

      ctx.restore();
    }
  }

  private drawRefusalBanner(canvasW: number, canvasH: number): void {
    const ctx = this.ctx;
    ctx.save();
    const bw = Math.min(480, canvasW - 24);
    const bh = 36;
    const bx = (canvasW - bw) / 2;
    const by = canvasH * 0.72;

    ctx.fillStyle = "rgba(220, 20, 60, 0.92)";
    ctx.beginPath();
    safeRoundRect(ctx, bx, by, bw, bh, 6);
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 13px system-ui, -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("HAZARD DETECTED — No Safe Footstep Recommendation", canvasW / 2, by + 23);
    ctx.restore();
  }

  private drawLegend(canvasW: number, canvasH: number): void {
    const ctx = this.ctx;
    const barH = 28;
    const y0 = canvasH - barH;

    ctx.save();
    ctx.fillStyle = "rgba(10, 12, 16, 0.88)";
    ctx.fillRect(0, y0, canvasW, barH);

    const classes = [
      { name: "Ground", color: "#228b22" },
      { name: "Obstacle", color: "#dc143c" },
      { name: "Vegetation", color: "#00c8c8" },
      { name: "Hole", color: "#8a2be2" },
      { name: "Uncertain", color: "#ffa500" },
    ];

    const itemW = canvasW / classes.length;
    ctx.font = "11px system-ui, -apple-system, sans-serif";
    ctx.textAlign = "left";

    for (let i = 0; i < classes.length; i++) {
      const cls = classes[i];
      const x = i * itemW + 6;
      const chipY = y0 + 7;

      ctx.fillStyle = cls.color;
      ctx.fillRect(x, chipY, 14, 14);
      ctx.strokeStyle = "#888888";
      ctx.lineWidth = 1;
      ctx.strokeRect(x, chipY, 14, 14);

      ctx.fillStyle = "#e0e0e0";
      ctx.fillText(cls.name, x + 18, y0 + 18);
    }
    ctx.restore();
  }
}
