/**
 * types.ts - Workspace 块类型定义 / Workspace block type definitions
 *
 * 工作区(拼图式分析工作台)的核心数据结构:数据集、序列块、模型块、可视化块、
 * 状态枚举等。所有 workspace 组件共享这套类型。 / Core data structures for
 * the puzzle-style analysis workbench: dataset, sequence/model/visualization
 * blocks, status enums, etc. Shared by all workspace components.
 *
 * 功能模块 / Modules:
 * - DatasetType / DATASET_OPTIONS / DATASET_LABELS: 数据集枚举 / dataset enum
 * - BlockType: 'sequence' | 'model' | 'visualization'
 * - SequenceStatus: 9 个状态(idle → completed/failed) / 9 statuses
 * - SequenceBlock / ModelBlock / VisualizationBlock: 三类块的接口 / 3 block interfaces
 * - 关联类型:对比配置、过滤器、布局 / related: comparison config, filter, layout
 *
 * 输入 / Inputs:
 * - 无(纯类型模块)/ None, pure type module
 *
 * 输出 / Outputs:
 * - 命名导出:类型、常量、Record / named exports: types, constants, records
 *
 * 数据流 / Data Flow:
 * 1. mockData.ts 引用此类型生成初始数据 / mockData.ts uses these types
 * 2. 各工作区组件 import 用于 props/state / components import for props/state
 * 3. 序列化到后端时按此结构编码 / Serialized to backend per these shapes
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: 无 / None
 * - 被调用 / Called by: src/components/workspace/*、src/pages/*、mockData.ts
 *
 * 使用示例 / Usage Example:
 *     import type { SequenceBlock, ModelBlock } from '@/components/workspace/types';
 *     const block: SequenceBlock = { id: uuidv7(), type: 'sequence', ... };
 *
 * 作者 / Author: 项目组 / Project Team
 * 版本 / Version: 1.0
 */
/**
 * Workspace Block Type Definitions
 * Defines the core data structures for the puzzle-style analysis workbench.
 */

// ==================== Dataset Types ====================

export type DatasetType = 'Human' | 'Plant' | '3Gen';

export const DATASET_OPTIONS: DatasetType[] = ['Human', 'Plant', '3Gen'];

export const DATASET_LABELS: Record<DatasetType, string> = {
  Human: 'Human',
  Plant: 'Plant',
  '3Gen': '3Gen',
};

// ==================== Block Types ====================

export type BlockType = 'sequence' | 'model' | 'visualization';

// ==================== Sequence Block ====================

export type SequenceStatus = 'idle' | 'validating' | 'validated' | 'submitting' | 'submitted' | 'queued' | 'processing' | 'completed' | 'failed';

export interface SequenceBlock {
  id: string;
  type: 'sequence';
  title: string;
  dataset: DatasetType;
  sequenceCount: number;
  sequence: string;
  status: SequenceStatus;
  jobId: string | null;
  resultSummary: {
    classification: any;
    attention: any;
    gcn: any;
  } | null;
  boundModelId: string | null;
}

// ==================== Model Block ====================

export type ModelStatus = 'available' | 'disabled' | 'loading';

export interface ModelBlock {
  id: string;
  type: 'model';
  title: string;
  modelName: string;
  status: ModelStatus;
  description: string;
  version: string;
}

// ==================== Visualization Block ====================

export type VizType =
  | 'classification'
  | 'attention'
  | 'attention-score'
  | 'gcn-graph'
  | 'gcn-message-passing'
  | 'integrated-gradients'
  | 'model-graph';

export type VizStatus = 'idle' | 'ready' | 'running' | 'completed' | 'failed';

export interface VisualizationBlock {
  id: string;
  type: 'visualization';
  title: string;
  vizType: VizType;
  status: VizStatus;
  boundSequenceId: string | null;
  boundModelId: string | null;
  params: Record<string, any>;
  result: any | null;
  autoRun: boolean;
}

// ==================== Union Block Type ====================

export type WorkspaceBlock = SequenceBlock | ModelBlock | VisualizationBlock;

// ==================== Canvas Zone ====================

export type CanvasZone = 'input' | 'model' | 'visualization';

// ==================== Visualization Metadata ====================

export interface VizTypeMeta {
  key: VizType;
  label: string;
  icon: string;
  description: string;
  defaultParams: Record<string, any>;
}

export const VIZ_TYPE_REGISTRY: VizTypeMeta[] = [
  {
    key: 'classification',
    label: 'Classification',
    icon: '🌳',
    description: 'Hierarchical classification tree showing model prediction paths.',
    defaultParams: {},
  },
  {
    key: 'attention',
    label: 'Attention',
    icon: '👁',
    description: 'Attention weight visualization highlighting important sequence positions.',
    defaultParams: { topX: 3, modificationType: 'all' },
  },
  {
    key: 'attention-score',
    label: 'Attention Score',
    icon: '📊',
    description: 'Per-class attention score distribution for all 12 modification types.',
    defaultParams: {},
  },
  {
    key: 'gcn-graph',
    label: 'GCN Graph',
    icon: '🔗',
    description: '3D graph structure of RNA sequence as processed by GCN.',
    defaultParams: {},
  },
  {
    key: 'gcn-message-passing',
    label: 'GCN Message Passing',
    icon: '📡',
    description: 'Message propagation through GCN layers for a target node.',
    defaultParams: { targetNodeIdx: 0 },
  },
  {
    key: 'integrated-gradients',
    label: 'Integrated Gradients',
    icon: '📈',
    description: 'Attribution scores showing which nodes contribute most to predictions.',
    defaultParams: { targetClassId: 0, topN: 10 },
  },
  {
    key: 'model-graph',
    label: 'Model Graph',
    icon: '🧠',
    description: 'Interactive data flow visualization of the model architecture.',
    defaultParams: {},
  },
];

// ==================== Binding Colors ====================

export interface BindingColor {
  border: string;
  bg: string;
  accent: string;
  tag: string;
}

export const BINDING_COLORS: BindingColor[] = [
  { border: '#e8917a', bg: '#fde8e0', accent: '#d4654a', tag: '#c04a30' },  // Coral
  { border: '#6bc26b', bg: '#e0f5e0', accent: '#4caf50', tag: '#388e3c' },  // Green
  { border: '#6aafe0', bg: '#ddeeff', accent: '#42a5f5', tag: '#1e88e5' },  // Blue
  { border: '#f0a050', bg: '#fff3e0', accent: '#ff9800', tag: '#f57c00' },  // Orange
  { border: '#b080d8', bg: '#f0e6f8', accent: '#9c27b0', tag: '#7b1fa2' },  // Purple
  { border: '#50c8c8', bg: '#e0f7f7', accent: '#00bcd4', tag: '#0097a7' },  // Teal
  { border: '#e07088', bg: '#fde0e6', accent: '#e91e63', tag: '#c2185b' },  // Pink
  { border: '#c8b040', bg: '#f8f4d8', accent: '#cddc39', tag: '#afb42b' },  // Lime
];

/**
 * Compute a stable binding color index for a model ID.
 * All blocks bound to the same model share the same color scheme.
 */
export function getBindingColorIndex(modelId: string | null): number {
  if (!modelId) return -1;
  // Simple hash from string to get a stable index
  let hash = 0;
  for (let i = 0; i < modelId.length; i++) {
    hash = ((hash << 5) - hash + modelId.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) % BINDING_COLORS.length;
}

// ==================== Task Execution ====================

export type TaskStatus = 'queued' | 'processing' | 'completed' | 'failed';

export interface TaskRecord {
  id: string;
  sequenceBlockId: string;
  status: TaskStatus;
  createdAt: number;
  completedAt: number | null;
  result: any | null;
  error: string | null;
}