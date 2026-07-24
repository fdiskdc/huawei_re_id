/**
 * api.config.ts - 后端 API 端点配置 / Backend API endpoint configuration
 *
 * 集中维护后端 API 端点(提交任务、轮询、IG、UMAP、注意力可视化、定位、对比等)
 * 与默认请求头;同时定义 Vite dev 代理目标。所有 API 相关的常量都集中在此。 /
 * Centralized backend API endpoints (submit task, polling, IG, UMAP, attention
 * viz, localization, comparison, etc.) and default headers; also defines the
 * Vite dev proxy target. All API-related constants live here.
 *
 * 功能模块 / Modules:
 * - DEFAULT_BASE_URL / BASE_URL: API 基础 URL(VITE_API_BASE_URL 可覆盖)/ base URL
 * - DEFAULT_HEADERS: 默认 Content-Type: application/json / default headers
 * - LEGACY_PREDICT_URL: 旧版 /api/predict 端点 / legacy predict endpoint
 * - ENDPOINTS: 所有 API 路径(/submit-task、/results/:jobId、/umap-data 等)/ all API paths
 * - PROXY_TARGET: Vite dev 代理目标 / Vite dev proxy target
 * - ApiConfig interface: 配置 TS 类型 / config TS type
 *
 * 输入 / Inputs:
 * - 环境变量:VITE_API_BASE_URL、VITE_LEGACY_PREDICT_URL、VITE_PROXY_TARGET / env vars
 *
 * 输出 / Outputs:
 * - 命名常量与 ENDPOINTS 对象 / named constants & ENDPOINTS object
 * - ApiConfig 类型 / ApiConfig type
 *
 * 数据流 / Data Flow:
 * 1. 模块加载时读 env,设置 BASE_URL / PROXY_TARGET / Set on load
 * 2. src/lib/api.ts 引用 ENDPOINTS 拼接 URL / api.ts uses ENDPOINTS
 * 3. Vite dev 代理 PROXY_TARGET 转发 / Vite dev proxies via PROXY_TARGET
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: vite/client(import.meta.env 类型)
 * - 被调用 / Called by: src/lib/api.ts、vite.config.ts、其他需要 URL 常量的模块
 *
 * 使用示例 / Usage Example:
 *     import { ENDPOINTS } from '@/config/api.config';
 *     fetch(ENDPOINTS.SUBMIT_TASK, { method: 'POST', body: ... });
 *
 * 作者 / Author: 项目组 / Project Team
 * 版本 / Version: 1.0
 */
/**
 * Centralized API configuration for DCPRES frontend
 * All API-related constants should be configured here
 */

/// <reference types="vite/client" />

// ==================== Base URL Configuration ====================

export const DEFAULT_BASE_URL = '/mrmodn/api/v1';
export const BASE_URL = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL) ?? DEFAULT_BASE_URL;

// ==================== Default Headers ====================

export const DEFAULT_HEADERS = {
    'Content-Type': 'application/json',
} as const;

// ==================== Legacy Endpoint Configuration ====================

export const DEFAULT_LEGACY_PREDICT_URL = `${DEFAULT_BASE_URL}/attention-visualization`;
export const LEGACY_PREDICT_URL = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_LEGACY_PREDICT_URL) ?? DEFAULT_LEGACY_PREDICT_URL;

// ==================== API Endpoints ====================

export const ENDPOINTS = {
    SUBMIT_TASK: `${BASE_URL}/submit-task`,
    GET_RESULT: (jobId: string) => `${BASE_URL}/results/${jobId}`,
    MODEL_GRAPH: `${BASE_URL}/model-graph`,
    INTEGRATED_GRADIENTS: `${BASE_URL}/integrated-gradients`,
    VISUALIZE_GCN_AGGREGATION: `${BASE_URL}/visualize-gcn-aggregation`,
    MODEL_COMPARISON: `${BASE_URL}/model-comparison`,
    MRMODN_CLASSIFICATION_HEATMAP: `${BASE_URL}/rgcnformer-classification-heatmap`,
    DATASET_COMPARISON: `${BASE_URL}/dataset-comparison-heatmap`,
    MRMODN_LOCALIZATION: `${BASE_URL}/rgcnformer-localization`,
    MRMODN_LOC_COMPARISON: `${BASE_URL}/rgcnformer-loc-comparison`,
    UMAP_DATA: `${BASE_URL}/umap-data`,
    UMAP_CORA_DATA: `${BASE_URL}/umap-cora-data`,
    PREDICT: LEGACY_PREDICT_URL,
    SAMPLE_SEQUENCE: `${BASE_URL}/sample-sequence`,
    ATTENTION_COMPARISON: `${BASE_URL}/attention-comparison`,
    ATTENTION_VISUALIZATION: `${BASE_URL}/attention-visualization`,
    REID_META: `${BASE_URL}/reid/meta`,
    REID_BATCH: (batchIndex: number, split: string) =>
        `${BASE_URL}/reid/batches/${batchIndex}?split=${encodeURIComponent(split)}`,
} as const;

// ==================== Vite Proxy Configuration ====================

export const DEFAULT_PROXY_TARGET = 'http://localhost:8005';
export const PROXY_TARGET = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_PROXY_TARGET) ?? DEFAULT_PROXY_TARGET;

// ==================== TypeScript Interface ====================

export interface ApiConfig {
    baseUrl: string;
    defaultHeaders: typeof DEFAULT_HEADERS;
    legacyPredictUrl: string;
    proxyTarget: string;
    endpoints: typeof ENDPOINTS;
}
