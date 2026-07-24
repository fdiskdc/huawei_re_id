/**
 * api.ts - 统一 API 客户端 / Unified API client
 *
 * 封装后端 HTTP 请求:基于 fetch 包装,统一错误处理、jobId 生成(uuidv7)、ResultData
 * 解析。支持提交任务、轮询结果、获取 IG / UMAP / 注意力 / 模型图等可视化数据。 /
 * Backend HTTP client: fetch-based wrapper with unified error handling, jobId
 * generation (uuidv7), and ResultData parsing. Supports task submission, result
 * polling, IG/UMAP/attention/model-graph visualization endpoints.
 *
 * 功能模块 / Modules:
 * - ResultData / SubmitTaskResponse 等 TS 类型 / TS types
 * - submitTask(sequence): 提交推理,返回 jobId / submit inference, return jobId
 * - getResult(jobId, onProgress): 轮询结果 / poll for result
 * - getModelGraph() / getIntegratedGradients() / getUMAPData() / getAttention(): 可视化数据 / viz data
 * - 错误处理:HTTP 错误、解析错误、超时 / error handling
 *
 * 输入 / Inputs:
 * - sequence: string - RNA 序列 / RNA sequence
 * - jobId: string - 任务 ID(由 uuidv7 生成)/ jobId (uuidv7)
 * - 可视化参数(targetClassId 等)/ viz params
 *
 * 输出 / Outputs:
 * - Promise<ResultData> - 后端结果 / backend result
 * - Promise<Blob/JSON> - 可视化数据 / viz data
 *
 * 数据流 / Data Flow:
 * 1. submitTask → POST ENDPOINTS.SUBMIT_TASK → 返回 jobId / submit, get jobId
 * 2. getResult 轮询 ENDPOINTS.GET_RESULT(jobId) → 拿到 ResultData / poll
 * 3. 用户切到可视化 tab → 调对应 viz 端点 / Switch tab → call viz endpoint
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: ../../config/api.config(ENDPOINTS、DEFAULT_HEADERS)、./uuidv7
 * - 被调用 / Called by: 各 pages(LocalizationViz、LocComparisonViz、CompareBarChart 等)
 *
 * 使用示例 / Usage Example:
 *     import { submitTask, getResult } from '@/lib/api';
 *     const { jobId } = await submitTask('ACGU...');
 *     const result = await getResult(jobId);
 *
 * 作者 / Author: 项目组 / Project Team
 * 版本 / Version: 1.0
 */
/**
 * Unified API client for DCPRES backend
 * Centralized API endpoint management
 */

// ==================== Import from Config ====================

import { ENDPOINTS, DEFAULT_HEADERS } from '../../config/api.config';
import { uuidv7 } from './uuidv7';

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
  integratedGradients?: any;
  gcnAggregation?: any;
  error?: string;
  errorType?: string;
  step?: string;
}

export interface ApiError {
  message: string;
  status?: number;
  detail?: string;
}

export type DatasetType = 'Human' | 'Plant' | '3Gen';

export interface SubmitTaskRequest {
  userId: string;
  rnaSequence: string;
  dataset?: DatasetType;
  datasetIndex?: number;
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

export interface CompareData {
  models: Array<{
    name: string;
    display_name: string;
    metrics: Record<string, number>;
  }>;
  metric_names: string[];
}

export interface MrmodnHeatmapData {
  model_name: string;
  classes: string[];
  metric_names: string[];
  data: Array<Record<string, string | number>>;
}

export interface DatasetComparisonData {
  dataset_names: string[];
  metric_names: string[];
  model_names: string[];
  row_labels: string[];
  data: Array<Record<string, string | number | null>>;
}

export interface MrmodnLocalizationData {
  model_name: string;
  classes: string[];
  class_names: string[];
  k_labels: string[];
  k_values: number[];
  heatmap: number[][];
  statistics: Array<{
    class: string;
    Mean: number;
    Median: number;
    Mode: number;
    Mode_Ratio: number;
    Sequence_Count: number;
    Min_Value: number;
    Max_Value: number;
    Standard_Deviation: number;
  }>;
}

export interface LocComparisonData {
  model_names: string[];
  k_labels: string[];
  k_values: number[];
  heatmap: number[][];
}

export interface UmapPoint {
  u1: number;
  u2: number;
  label: string;
  group: string;
  seq: string;
  probs: number[];
}

export interface DensityContour {
  level: number;
  polygons: number[][][];
}

export interface UmapMetadata {
  n_per_class: number;
  total_points: number;
  valid_classes: number[];
  color_map: Record<string, string>;
  group_colors: Record<string, string>;
  label_names: string[];
  group_mapping: Record<string, number[]>;
  label_num_samples: Record<string, number>;
  subsampled?: boolean;
}

export interface UmapData {
  points: UmapPoint[];
  density_contours: Record<string, DensityContour[]>;
  metadata: UmapMetadata;
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
 * Generate a time-sortable UUID v7 job identifier
 */
export function generateJobId(): string {
  return uuidv7();
}

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
      ...DEFAULT_HEADERS,
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
      ...DEFAULT_HEADERS,
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
      ...DEFAULT_HEADERS,
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

/**
 * Fetch model comparison data
 */
export async function fetchModelComparison(): Promise<CompareData> {
  const response = await fetch(ENDPOINTS.MODEL_COMPARISON);

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

/**
 * Fetch DCPRES classification heatmap data
 */
export async function fetchMrmodnHeatmap(): Promise<MrmodnHeatmapData> {
  const response = await fetch(ENDPOINTS.MRMODN_CLASSIFICATION_HEATMAP);

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

/**
 * Fetch dataset comparison heatmap data
 */
export async function fetchDatasetComparison(): Promise<DatasetComparisonData> {
  const response = await fetch(ENDPOINTS.DATASET_COMPARISON);

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

/**
 * Fetch DCPRES localization data
 */
export async function fetchMrmodnLocalization(): Promise<MrmodnLocalizationData> {
  const response = await fetch(ENDPOINTS.MRMODN_LOCALIZATION);

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

/**
 * Fetch DCPRES localization comparison data
 */
export async function fetchMrmodnLocComparison(): Promise<LocComparisonData> {
  const response = await fetch(ENDPOINTS.MRMODN_LOC_COMPARISON);

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

export async function fetchUmapData(): Promise<UmapData> {
  const response = await fetch(ENDPOINTS.UMAP_DATA);

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

export async function fetchCoraUmapData(): Promise<UmapData> {
  const response = await fetch(ENDPOINTS.UMAP_CORA_DATA);

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

/**
 * Fetch a random sample sequence for workspace input block
 */
export async function fetchSampleSequence(): Promise<{ sequence: string }> {
  const response = await fetch(ENDPOINTS.SAMPLE_SEQUENCE);

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

// ==================== Attention Visualization Types ====================

export interface AttentionComparisonSample {
  index: number;
  models: Record<string, {
    attention: number[][];
    class_indices: number[];
    class_names: string[];
    true_sites: number[];
  }>;
}

export interface AttentionComparisonData {
  samples: AttentionComparisonSample[];
  class_names: string[];
  model_names: string[];
  available_sequence_ids?: number[];
}

export interface AttentionClassData {
  index: number;
  name: string;
  probability: number;
  attention: number[];
}

export interface AttentionVisualizationData {
  sequence_length: number;
  left_padding: number;
  classes: AttentionClassData[];
  class_names: string[];
}

// ==================== Attention Visualization Functions ====================

/**
 * Fetch pre-computed attention comparison data for 4 models
 */
export async function fetchAttentionComparison(sequenceId?: number): Promise<AttentionComparisonData> {
  const url = sequenceId === undefined
    ? ENDPOINTS.ATTENTION_COMPARISON
    : `${ENDPOINTS.ATTENTION_COMPARISON}?sequence_id=${encodeURIComponent(sequenceId)}`;
  const response = await fetch(url);

  if (!response.ok) {
    throw await createApiError(response);
  }

  return response.json();
}

/**
 * Fetch attention visualization for a user-submitted sequence
 */
export async function fetchAttentionVisualization(request: { rnaSequence: string }): Promise<AttentionVisualizationData> {
  const response = await fetch(ENDPOINTS.ATTENTION_VISUALIZATION, {
    method: 'POST',
    headers: {
      ...DEFAULT_HEADERS,
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
      ...DEFAULT_HEADERS,
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
