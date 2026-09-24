/**
 * StepView Web — 2D Footstep Candidate Engine (TypeScript)
 *
 * Direct browser implementation of the StepView footstep proposal pipeline:
 * - Computes hazard distance map via two-pass distance transform
 * - Enforces minimum obstacle clearance
 * - Scales footprint dimensions with perspective lookahead
 * - Evaluates multi-factor candidate safety scores
 * - Applies Non-Maximum Suppression (NMS) to select top footstep placements
 */

import { FootstepCandidate, ProposalResult, TerrainClass } from "./types.js";

export interface FootstepEngineConfig {
  minClearancePx?: number;
  idealClearancePx?: number;
  minGroundSupportRatio?: number;
  lookaheadMinYRatio?: number;
  lookaheadMaxYRatio?: number;
  nearFootprintSize?: [number, number]; // [width, height] in px
  farFootprintSize?: [number, number];
  topK?: number;
}

export class FootstepProposalEngineWeb {
  public minClearancePx: number;
  public idealClearancePx: number;
  public minGroundSupportRatio: number;
  public lookaheadMinYRatio: number;
  public lookaheadMaxYRatio: number;
  public nearFootprintSize: [number, number];
  public farFootprintSize: [number, number];
  public topK: number;

  constructor(config: FootstepEngineConfig = {}) {
    this.minClearancePx = config.minClearancePx ?? 10.0;
    this.idealClearancePx = config.idealClearancePx ?? 40.0;
    this.minGroundSupportRatio = config.minGroundSupportRatio ?? 0.70;
    this.lookaheadMinYRatio = config.lookaheadMinYRatio ?? 0.35;
    this.lookaheadMaxYRatio = config.lookaheadMaxYRatio ?? 0.95;
    this.nearFootprintSize = config.nearFootprintSize ?? [56, 34];
    this.farFootprintSize = config.farFootprintSize ?? [28, 16];
    this.topK = config.topK ?? 5;
  }

  /**
   * Fast two-pass Chamfer distance transform for hazard clearance.
   */
  public computeHazardDistanceMap(
    classMask: Uint8Array,
    width: number,
    height: number
  ): Float32Array {
    const size = width * height;
    const dist = new Float32Array(size);
    const INF = 1e5;

    // Pass 0: Init hazard cells to 0, non-hazard to INF
    for (let i = 0; i < size; i++) {
      const cls = classMask[i];
      if (
        cls === TerrainClass.OBSTACLE ||
        cls === TerrainClass.HOLE ||
        cls === TerrainClass.UNCERTAIN_SURFACE
      ) {
        dist[i] = 0.0;
      } else {
        dist[i] = INF;
      }
    }

    // Pass 1: Top-left to bottom-right
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        let d = dist[idx];
        if (d === 0.0) continue;

        if (x > 0) d = Math.min(d, dist[idx - 1] + 1.0);
        if (y > 0) d = Math.min(d, dist[idx - width] + 1.0);
        if (x > 0 && y > 0) d = Math.min(d, dist[idx - width - 1] + 1.414);
        if (x + 1 < width && y > 0) d = Math.min(d, dist[idx - width + 1] + 1.414);

        dist[idx] = d;
      }
    }

    // Pass 2: Bottom-right to top-left
    for (let y = height - 1; y >= 0; y--) {
      for (let x = width - 1; x >= 0; x--) {
        const idx = y * width + x;
        let d = dist[idx];
        if (d === 0.0) continue;

        if (x + 1 < width) d = Math.min(d, dist[idx + 1] + 1.0);
        if (y + 1 < height) d = Math.min(d, dist[idx + width] + 1.0);
        if (x + 1 < width && y + 1 < height) d = Math.min(d, dist[idx + width + 1] + 1.414);
        if (x > 0 && y + 1 < height) d = Math.min(d, dist[idx + width - 1] + 1.414);

        dist[idx] = d;
      }
    }

    return dist;
  }

  /**
   * Perspective scaling for expected footprint size at vertical coordinate y.
   */
  public getFootprintSize(y: number, height: number): [number, number] {
    const yNorm = Math.min(Math.max(y / height, this.lookaheadMinYRatio), this.lookaheadMaxYRatio);
    const t = (yNorm - this.lookaheadMinYRatio) / (this.lookaheadMaxYRatio - this.lookaheadMinYRatio + 1e-6);

    const [wNear, hNear] = this.nearFootprintSize;
    const [wFar, hFar] = this.farFootprintSize;

    const w = wFar + t * (wNear - wFar);
    const h = hFar + t * (hNear - hFar);
    return [w, h];
  }

  /**
   * Evaluate safety and score of an elliptical footstep candidate at (cx, cy).
   */
  public evaluateCandidate(
    cx: number,
    cy: number,
    classMask: Uint8Array,
    distMap: Float32Array,
    confMap: Float32Array,
    width: number,
    height: number
  ): FootstepCandidate {
    const [fpW, fpH] = this.getFootprintSize(cy, height);
    const rx = fpW / 2.0;
    const ry = fpH / 2.0;

    const xMin = Math.max(0, Math.floor(cx - rx));
    const xMax = Math.min(width - 1, Math.ceil(cx + rx));
    const yMin = Math.max(0, Math.floor(cy - ry));
    const yMax = Math.min(height - 1, Math.ceil(cy + ry));

    let totalInside = 0;
    let groundCount = 0;
    let hazardCount = 0;
    let minClearance = 1e5;
    let modelConfSum = 0;

    for (let y = yMin; y <= yMax; y++) {
      for (let x = xMin; x <= xMax; x++) {
        const dx = (x - cx) / rx;
        const dy = (y - cy) / ry;
        if (dx * dx + dy * dy <= 1.0) {
          totalInside++;
          const idx = y * width + x;
          const cls = classMask[idx];

          if (cls === TerrainClass.GROUND) {
            groundCount++;
          } else if (
            cls === TerrainClass.OBSTACLE ||
            cls === TerrainClass.HOLE ||
            cls === TerrainClass.UNCERTAIN_SURFACE
          ) {
            hazardCount++;
          }

          minClearance = Math.min(minClearance, distMap[idx]);
          modelConfSum += confMap[idx];
        }
      }
    }

    if (totalInside === 0) {
      return {
        rank: 0,
        x_center: cx,
        y_center: cy,
        width: fpW,
        height: fpH,
        score: 0.0,
        confidence: 0.0,
        ground_confidence: 0.0,
        model_confidence: 0.0,
        obstacle_clearance_px: 0.0,
        compactness: 0.0,
        area_px: 0.0,
        distance_from_bottom_norm: 0.0,
        is_valid: false,
        rejection_reason: "empty_footprint",
      };
    }

    const groundRatio = groundCount / totalInside;
    const avgConf = modelConfSum / totalInside;
    const clearance = minClearance === 1e5 ? 0.0 : minClearance;

    // Hard rejection checks
    let isValid = true;
    let rejectionReason: string | null = null;

    if (hazardCount > 0) {
      isValid = false;
      rejectionReason = "hazard_overlap";
    } else if (groundRatio < this.minGroundSupportRatio) {
      isValid = false;
      rejectionReason = "insufficient_ground";
    } else if (clearance < this.minClearancePx) {
      isValid = false;
      rejectionReason = "insufficient_clearance";
    }

    // Normalized components
    const clearanceNorm = Math.min(1.0, clearance / this.idealClearancePx);
    const distBottomNorm = Math.min(1.0, Math.max(0.0, 1.0 - cy / height));
    const compactness = 1.0;

    // Multi-factor composite heuristic score
    const score =
      0.35 * groundRatio +
      0.25 * clearanceNorm +
      0.20 * compactness +
      0.10 * avgConf +
      0.10 * (1.0 - distBottomNorm); // Favor closer reachable steps

    return {
      rank: 0,
      x_center: Math.round(cx * 10) / 10,
      y_center: Math.round(cy * 10) / 10,
      width: Math.round(fpW * 10) / 10,
      height: Math.round(fpH * 10) / 10,
      score: Math.round(score * 1000) / 1000,
      confidence: Math.round(groundRatio * 1000) / 1000,
      ground_confidence: Math.round(groundRatio * 1000) / 1000,
      model_confidence: Math.round(avgConf * 1000) / 1000,
      obstacle_clearance_px: Math.round(clearance * 10) / 10,
      compactness: 1.0,
      area_px: Math.round(Math.PI * rx * ry * 10) / 10,
      distance_from_bottom_norm: Math.round(distBottomNorm * 1000) / 1000,
      is_valid: isValid,
      rejection_reason: rejectionReason,
    };
  }

  /**
   * Apply Non-Maximum Suppression (NMS) to eliminate duplicate/overlapping candidates.
   */
  public applyNMS(candidates: FootstepCandidate[]): FootstepCandidate[] {
    const sorted = [...candidates].sort((a, b) => b.score - a.score);
    const selected: FootstepCandidate[] = [];

    for (const cand of sorted) {
      let keep = true;
      for (const sel of selected) {
        const dx = cand.x_center - sel.x_center;
        const dy = cand.y_center - sel.y_center;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const minDist = Math.min(cand.width, sel.width) * 0.70;

        if (dist < minDist) {
          keep = false;
          break;
        }
      }

      if (keep) {
        cand.rank = selected.length + 1;
        selected.push(cand);
        if (selected.length >= this.topK) break;
      }
    }

    return selected;
  }

  /**
   * Main proposal pipeline: extracts distance map, samples ground grid, and returns top K footsteps.
   */
  public proposeFootsteps(
    classMask: Uint8Array,
    confMap: Float32Array,
    width: number,
    height: number
  ): ProposalResult {
    const t0 = performance.now();
    const distMap = this.computeHazardDistanceMap(classMask, width, height);

    const candidates: FootstepCandidate[] = [];
    const stepY = Math.max(16, Math.floor(height / 18));
    const stepX = Math.max(16, Math.floor(width / 18));

    const yStart = Math.floor(height * this.lookaheadMinYRatio);
    const yEnd = Math.floor(height * this.lookaheadMaxYRatio);

    // Sample candidate points across ground zone
    for (let y = yStart; y <= yEnd; y += stepY) {
      for (let x = Math.floor(width * 0.15); x <= Math.floor(width * 0.85); x += stepX) {
        const idx = y * width + x;
        if (classMask[idx] === TerrainClass.GROUND) {
          const cand = this.evaluateCandidate(x, y, classMask, distMap, confMap, width, height);
          candidates.push(cand);
        }
      }
    }

    const validCandidates = candidates.filter((c) => c.is_valid);
    const topCandidates = this.applyNMS(validCandidates);
    const latencyMs = performance.now() - t0;

    return {
      topCandidates,
      allCandidates: candidates,
      validCount: validCandidates.length,
      evaluatedCount: candidates.length,
      latencyMs,
    };
  }
}
