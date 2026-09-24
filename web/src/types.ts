/**
 * StepView Web — Core Types & Schema
 */

export enum TerrainClass {
  BACKGROUND = 0,
  GROUND = 1,
  OBSTACLE = 2,
  VEGETATION = 3,
  HOLE = 4,
  UNCERTAIN_SURFACE = 5,
}

export const CLASS_NAMES: Record<TerrainClass, string> = {
  [TerrainClass.BACKGROUND]: "Background",
  [TerrainClass.GROUND]: "Ground",
  [TerrainClass.OBSTACLE]: "Obstacle",
  [TerrainClass.VEGETATION]: "Vegetation",
  [TerrainClass.HOLE]: "Hole/Drop",
  [TerrainClass.UNCERTAIN_SURFACE]: "Uncertain",
};

export const CLASS_PALETTE: Record<TerrainClass, [number, number, number]> = {
  [TerrainClass.BACKGROUND]: [0, 0, 0],          // Black
  [TerrainClass.GROUND]: [34, 139, 34],          // Forest Green
  [TerrainClass.OBSTACLE]: [220, 20, 60],        // Crimson Red
  [TerrainClass.VEGETATION]: [0, 200, 200],      // Cyan / Teal
  [TerrainClass.HOLE]: [138, 43, 226],           // Blue Violet
  [TerrainClass.UNCERTAIN_SURFACE]: [255, 165, 0], // Amber / Orange
};

export interface FootstepCandidate {
  rank: number;
  x_center: number;
  y_center: number;
  width: number;
  height: number;
  score: number;
  confidence: number;
  ground_confidence: number;
  model_confidence: number;
  obstacle_clearance_px: number;
  compactness: number;
  area_px: number;
  distance_from_bottom_norm: number;
  is_valid: boolean;
  rejection_reason: string | null;
}

export interface SegmentationResult {
  classMask: Uint8Array;
  confidenceMap: Float32Array;
  probabilities: Float32Array;
  width: number;
  height: number;
  latencyMs: number;
}

export interface ProposalResult {
  topCandidates: FootstepCandidate[];
  allCandidates: FootstepCandidate[];
  validCount: number;
  evaluatedCount: number;
  latencyMs: number;
}
