/**
 * Unified API client for RGCNFormer backend
 * Centralized API endpoint management
 */

// ==================== API Endpoints ====================
const API_BASE_URL = '/rgcnformer/api/v1';

const REID_API_BASE_URL = `${API_BASE_URL}/reid`;

const ENDPOINTS = {
  // Task submission
  SUBMIT_TASK: `${API_BASE_URL}/submit-task`,

  // ReID endpoints
  REID_META: `${REID_API_BASE_URL}/meta`,
  REID_BATCH: (batchIndex: number, split: ReidSplit) =>
    `${REID_API_BASE_URL}/batches/${batchIndex}?split=${encodeURIComponent(split)}`,
  
  // Result retrieval
  GET_RESULT: (jobId: string) => `${API_BASE_URL}/results/${jobId}`,
  
  // Visualizations
  MODEL_GRAPH: `${API_BASE_URL}/model-graph`,
  INTEGRATED_GRADIENTS: `${API_BASE_URL}/integrated-gradients`,
  VISUALIZE_GCN_AGGREGATION: `${API_BASE_URL}/visualize-gcn-aggregation`,
  
  // Legacy endpoint (hardcoded localhost - should be updated)
  PREDICT: 'http://localhost:5000/rgcnformer/api/predict',
} as const;

// ==================== Type Definitions ====================

export interface ResultData {
  jobId: string;
  status: 'completed' | 'processing' | 'failed' | 'unknown' | 'RETRY';
  classification?: any;
  attention?: {
    sequence: string;
    weights: Array<{
      index: number;
      type: string;
      score: number;
    }>;
  };
  gcn?: {
    nodes: Array<{
      id: string;
      label: string;
      data: {
        index: number;
        type: string;
        name: string;
      };
    }>;
    edges: Array<{
      source: string;
      target: string;
    }>;
  };
  error?: string;
  errorType?: string;
  step?: string;
}

export interface ApiError {
  message: string;
  status?: number;
  detail?: string;
}

export interface SubmitTaskRequest {
  userId: string;
  rnaSequence: string;
  jobId?: string;
}

export interface SubmitTaskResponse {
  jobId: string;
  status: string;
  message?: string;
  classification?: any;
  attention?: any;
  gcn?: any;
  integratedGradients?: any;
}

export interface IntegratedGradientsRequest {
  rnaSequence: string;
  targetClassId: number;
}

export interface GcnAggregationRequest {
  rnaSequence: string;
  targetNodeIdx: number;
}

export type ReidStage = 'position_embedding' | 'transformer_0' | 'transformer_1' | 'classifier';
export type ReidSplit = 'train' | 'test' | 'val';

export interface ReidHeatmap {
  values: number[][];
  rawMin: number;
  rawMax: number;
}

export interface ReidSample {
  sampleId: string;
  pid: number;
  camera: number;
  modality: 'visible' | 'infrared';
  relativePath: string;
  imageUrl: string;
  prediction: { classIndex: number; pid: number; score: number };
  heatmaps: Record<ReidStage, ReidHeatmap>;
}

export interface ReidBatchResponse {
  batchId: string;
  batchIndex: number;
  batchSize: number;
  totalBatches: number;
  gridShape: [number, number];
  displaySize: [number, number];
  stages: ReidStage[];
  samples: ReidSample[];
}

export interface ReidMetaResponse {
  status: 'ready';
  device: 'cpu';
  torchThreads: number;
  batchSize: number;
  gridShape: [number, number];
  displaySize: [number, number];
  stages: ReidStage[];
  defaultSplit: ReidSplit;
  splits: ReidSplit[];
  totalBatches: Record<ReidSplit, number>;
  modelConfig: Record<string, number>;
  checkpoint: string;
}

// ==================== Utility Functions ====================

/**
 * Create standardized error object
 */
async function createApiError(response: Response): Promise<ApiError> {
  const error: ApiError = {
    message: `HTTP ${response.status}: ${response.statusText}`,
    status: response.status,
  };

  try {
    const errorData = await response.json();
    error.detail = errorData.error || errorData.detail;
  } catch {
    // Ignore JSON parsing errors for error responses
  }

  return error;
}

/**
 * Check if result data indicates processing is still in progress
 */
export function isProcessing(data: ResultData | undefined): boolean {
  if (!data) return false;
  return data.status === 'processing' || data.status === 'unknown' || data.status === 'RETRY';
}

/**
 * Check if result data indicates success
 */
export function isCompleted(data: ResultData | undefined): boolean {
  if (!data) return false;
  return data.status === 'completed' || !!data.classification;
}

/**
 * Check if result data indicates failure
 */
export function isFailed(data: ResultData | undefined): boolean {
  if (!data) return false;
  return data.status === 'failed' || !!data.error;
}

/**
 * Get error message from result data
 */
export function getErrorMessage(data: ResultData | undefined): string | null {
  if (!data) return null;
  if (data.error) return data.error;
  return null;
}

// ==================== API Functions ====================

/**
 * Submit a new task for processing
 */
export async function submitTask(request: SubmitTaskRequest): Promise<SubmitTaskResponse> {
  const response = await fetch(ENDPOINTS.SUBMIT_TASK, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

/**
 * Fetch result by job ID
 */
export async function fetchResult(jobId: string): Promise<ResultData> {
  const response = await fetch(ENDPOINTS.GET_RESULT(jobId));

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

/**
 * Fetch model graph data
 */
export async function fetchModelGraph(): Promise<any> {
  const response = await fetch(ENDPOINTS.MODEL_GRAPH);

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

/**
 * Fetch integrated gradients data
 */
export async function fetchIntegratedGradients(request: IntegratedGradientsRequest): Promise<any> {
  const response = await fetch(ENDPOINTS.INTEGRATED_GRADIENTS, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

/**
 * Fetch GCN aggregation visualization data
 */
export async function fetchGcnAggregation(request: GcnAggregationRequest): Promise<any> {
  const response = await fetch(ENDPOINTS.VISUALIZE_GCN_AGGREGATION, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

/**
 * Legacy predict function (hardcoded localhost - consider updating)
 */
export async function predict(request: any): Promise<any> {
  const response = await fetch(ENDPOINTS.PREDICT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

export async function fetchReidMeta(signal?: AbortSignal): Promise<ReidMetaResponse> {
  const response = await fetch(ENDPOINTS.REID_META, { signal });
  if (!response.ok) throw await createApiError(response);
  return response.json();
}

function validateReidBatch(batch: ReidBatchResponse): ReidBatchResponse {
  if (batch.batchSize !== 4 || batch.samples.length !== 4) {
    throw new Error('ReID response must contain exactly four samples');
  }
  if (batch.gridShape[0] !== 9 || batch.gridShape[1] !== 5 || batch.stages.length !== 4) {
    throw new Error('ReID response has an unexpected stage or grid shape');
  }
  for (const sample of batch.samples) {
    for (const stage of batch.stages) {
      const values = sample.heatmaps[stage]?.values;
      if (!values || values.length !== 9 || values.some((row) => row.length !== 5)) {
        throw new Error(`Invalid heatmap shape for ${stage}`);
      }
    }
  }
  return batch;
}

export async function fetchReidBatch(
  batchIndex: number,
  split: ReidSplit,
  signal?: AbortSignal,
): Promise<ReidBatchResponse> {
  const response = await fetch(ENDPOINTS.REID_BATCH(batchIndex, split), { signal });
  if (!response.ok) throw await createApiError(response);
  return validateReidBatch(await response.json());
}

// ==================== Export Constants ====================
export { ENDPOINTS };
