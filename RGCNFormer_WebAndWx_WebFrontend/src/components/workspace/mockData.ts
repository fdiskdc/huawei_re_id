/**
 * mockData.ts - Workspace 模拟数据 / Workspace mock data
 *
 * 工作区在开发期(无后端 / 演示模式)使用的模拟数据:默认模型、示例序列、示例
 * 可视化结果、generateId 工具等。 / Mock data for the workbench during dev/demo
 * (no backend): default models, sample sequences, sample visualization results,
 * generateId helper, etc.
 *
 * 功能模块 / Modules:
 * - generateId(prefix): 基于时间戳 + 计数器的 ID / timestamp + counter ID
 * - DEFAULT_MODELS: 默认模型块(mRModN / GCN / Baseline)/ default model blocks
 * - DEFAULT_SEQUENCE: 示例 RNA 序列 / sample RNA sequence
 * - SAMPLE_VISUALIZATION_DATA: 示例注意力 / 定位数据 / sample attention/loc data
 *
 * 输入 / Inputs:
 * - 无(纯数据模块)/ None, pure data module
 *
 * 输出 / Outputs:
 * - generateId(prefix): string - 唯一 ID / unique ID
 * - DEFAULT_MODELS: ModelBlock[] - 默认模型 / default models
 * - 其他:SequenceBlock、VisualizationBlock 等示例 / other sample blocks
 *
 * 数据流 / Data Flow:
 * 1. 工作区初始化时如无真实数据,使用本模块的 mock / Init with mock if no real data
 * 2. 用户切换数据源时由真实数据替换 mock / Switch to real data when available
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: ./types(类型导入)
 * - 被调用 / Called by: 工作区初始化、各 workspace 组件 / workspace init, components
 *
 * 使用示例 / Usage Example:
 *     import { DEFAULT_MODELS, generateId } from '@/components/workspace/mockData';
 *     const id = generateId('seq');
 *
 * 作者 / Author: 项目组 / Project Team
 * 版本 / Version: 1.0
 */
/**
 * Mock Data & Sample Results for Workspace
 * Centralized mock data, sample results, and default visualization data.
 */

import type { SequenceBlock, ModelBlock, VisualizationBlock, DatasetType } from './types';

let _idCounter = 0;
export const generateId = (prefix: string): string => {
  _idCounter++;
  return `${prefix}_${Date.now()}_${_idCounter}`;
};

// ==================== Default Models ====================

export const DEFAULT_MODELS: ModelBlock[] = [
  {
    id: 'model_mrmodn',
    type: 'model',
    title: 'DCPRES',
    modelName: 'DCPRES',
    status: 'available',
    description:
      'Relation-aware Graph Convolutional Network with Transformer for RNA modification prediction. Supports 12 RNA modification types across A, C, G, U nucleotides.',
    version: 'v1.0',
  },
  {
    id: 'model_gcn',
    type: 'model',
    title: 'GCN',
    modelName: 'GCN',
    status: 'available',
    description:
      'Graph Convolutional Network for RNA modification site prediction. Leverages local graph structure of RNA sequences.',
    version: 'v0.1',
  },
  {
    id: 'model_kmeans',
    type: 'model',
    title: 'K-Means',
    modelName: 'K-Means',
    status: 'available',
    description:
      'K-Means clustering for unsupervised RNA modification pattern discovery and grouping.',
    version: 'v0.1',
  },
  {
    id: 'model_dscps',
    type: 'model',
    title: 'DSCPS',
    modelName: 'DSCPS',
    status: 'available',
    description:
      'Deep Semi-Supervised Clustering based on Pairwise Constraints and Sample Similarity',
    version: 'v1.0',
  },
  {
    id: 'model_procse',
    type: 'model',
    title: 'ProCSE',
    modelName: 'ProCSE',
    status: 'available',
    description:
      'Progressive Contrastive Structural Entropy for Graph Clustering',
    version: 'v1.0',
  },
];

// ==================== Sample Sequence ====================

export const SAMPLE_SEQUENCE =
  'TCAGGAGTTCGAGACCAGCCTGATCAACATGACGAAACCCTATCTCTACTAAAAATACAAAAATTAGCCGGGCGTGGTGGCATGCGCCTGTAGTCTCAGCTACTTGGGAGGCTGAAGCAGGAGAATCGTTTGAACCCAGGAGGCAGAGGTTGCAGTGAGCCGAGATCGTGCCACTGCACTCCAGCCTGGGTGACACAGCGAGACTCTGTCTCAAAAAAATAAAAATAAAAAAATAAATAAATAACCTTTAATTTAGTGAGACTTCATATAGAATTGTTTTAATGTTTAATATAGACCATTTGTTTTAGGTGAATTTAACAATTTCATACTGTGATTAAGATTAATTTCTTTTTCTGACTTCTACCAGAAAGCAGGAATTATGTTTCAAATGGACAATCATTTACCAAACCTTGTTAATCTGAATGAAGATCCACAACTATCTGAGATGCTGCTATATATGATAAAAGAAGGAACAACTACAGTTGGAAAGTATAAACCAAACTCAAGCCATGATATTCAGTTATCTGGGGTGCTGATTGCTGATGATCATTGGTATGTTAATCCTCTAAAAAAAAAGAAAAGGCACCTGTTCTATATCTTGATAACATGTGGTTTCCTTCATATGGCATATTCGTTGATACTGATCGTTTGGTAGAATTCTTCAAACCCATTGTTTAGTCAGGAAAAACATACATTCTGAGTGTGTTATAAGGATGATAGGTCAGTTACTCTCAATATAAAGTACAGTGTAATGCTCTCTCTGTTTTTGTTTTGGCATACTTGATCTGTTGATTGAAGAATAATTTATTTTCTTGCAATTATAATGATGCACATGCAAGTAAACTATCTATCTTACATAACAGAATTTTTGGTTGGATTGACCAATTTAAAAATGTTACTTTATGTGAATTTTGTTCATATGAATGGAATACTTGTATATATTGTTGGAATGATAGCGTATGTAAACTTTTTTGACTCTGCATTGTGTTTCCAAGATTTGT';

// ==================== Mock Classification Result ====================

export const MOCK_CLASSIFICATION = {
  name: 'Root',
  isPredicted: false,
  children: [
    {
      name: 'A',
      isPredicted: false,
      children: [
        {
          name: 'Am',
          isPredicted: true,
          children: [],
        },
        {
          name: 'Atol',
          isPredicted: false,
          children: [],
        },
        {
          name: 'm1A',
          isPredicted: false,
          children: [],
        },
        {
          name: 'm6A',
          isPredicted: true,
          children: [],
        },
        {
          name: 'm6Am',
          isPredicted: false,
          children: [],
        },
      ],
    },
    {
      name: 'C',
      isPredicted: false,
      children: [
        {
          name: 'Cm',
          isPredicted: false,
          children: [],
        },
        {
          name: 'ac4C',
          isPredicted: false,
          children: [],
        },
        {
          name: 'm5C',
          isPredicted: true,
          children: [],
        },
      ],
    },
    {
      name: 'G',
      isPredicted: false,
      children: [
        {
          name: 'Gm',
          isPredicted: false,
          children: [],
        },
        {
          name: 'm7G',
          isPredicted: false,
          children: [],
        },
      ],
    },
    {
      name: 'U',
      isPredicted: false,
      children: [
        {
          name: 'Tm',
          isPredicted: false,
          children: [],
        },
        {
          name: 'Y',
          isPredicted: false,
          children: [],
        },
      ],
    },
  ],
};

// ==================== Mock Attention Result ====================

export const MOCK_ATTENTION = (seq: string) => ({
  sequence: seq,
  weights: Array.from({ length: 20 }, (_, i) => ({
    index: Math.floor(Math.random() * seq.length),
    type: ['A', 'C', 'G', 'U'][i % 4],
    score: Math.random(),
  })),
});

// ==================== Mock GCN Result ====================

export const MOCK_GCN = (seq: string) => {
  const nodes = Array.from({ length: Math.min(seq.length, 50) }, (_, i) => ({
    id: `node_${i}`,
    label: `${seq[i] || 'N'}${i}`,
    data: {
      index: i,
      type: seq[i] || 'N',
      name: `Nucleotide ${i}`,
    },
  }));

  const edges: Array<{ source: string; target: string }> = [];
  // Backbone edges
  for (let i = 0; i < nodes.length - 1; i++) {
    edges.push({ source: `node_${i}`, target: `node_${i + 1}` });
  }
  // Some pairing edges
  for (let i = 0; i < Math.min(10, nodes.length); i++) {
    const j = Math.min(i + 4, nodes.length - 1);
    if (i !== j) {
      edges.push({ source: `node_${i}`, target: `node_${j}` });
    }
  }

  return { nodes, edges };
};

// ==================== Mock GCN Aggregation Result ====================

export const MOCK_GCN_AGGREGATION = (seq: string) => {
  const nodeCount = Math.min(seq.length, 50);
  const nodes = Array.from({ length: nodeCount }, (_, i) => ({
    id: `node_${i}`,
    data: {
      index: i,
      type: seq[i] || 'N',
      name: `Nucleotide ${i}`,
    },
  }));

  const edges: Array<{ source: string; target: string }> = [];
  for (let i = 0; i < nodeCount - 1; i++) {
    edges.push({ source: `node_${i}`, target: `node_${i + 1}` });
  }
  for (let i = 0; i < Math.min(10, nodeCount); i++) {
    const j = Math.min(i + 4, nodeCount - 1);
    if (i !== j) {
      edges.push({ source: `node_${i}`, target: `node_${j}` });
    }
  }

  const targetNode = 0;
  const aggregationData = Array.from({ length: 3 }, (_, layer) => ({
    layer,
    messages: edges.slice(0, 5).map((e) => ({
      from: parseInt(e.source.split('_')[1]),
      strength: Math.random(),
    })),
  }));

  return { targetNode, nodes, edges, aggregationData };
};

// ==================== Mock Integrated Gradients Result ====================

export const MOCK_INTEGRATED_GRADIENTS = (seq: string) => {
  const nodeCount = Math.min(seq.length, 50);
  const nodes = Array.from({ length: nodeCount }, (_, i) => ({
    id: `node_${i}`,
    label: `${seq[i] || 'N'}${i}`,
    data: {
      index: i,
      type: seq[i] || 'N',
      name: `Nucleotide ${i}`,
      attributionScore: (Math.random() - 0.5) * 2,
    },
  }));

  const edges: Array<{ source: string; target: string }> = [];
  for (let i = 0; i < nodeCount - 1; i++) {
    edges.push({ source: `node_${i}`, target: `node_${i + 1}` });
  }
  for (let i = 0; i < Math.min(10, nodeCount); i++) {
    const j = Math.min(i + 4, nodeCount - 1);
    if (i !== j) {
      edges.push({ source: `node_${i}`, target: `node_${j}` });
    }
  }

  return { nodes, edges };
};

// ==================== Task Simulation ====================

export interface MockTaskRunner {
  start: (sequenceBlockId: string) => string;
  onResult: (callback: (result: { sequenceBlockId: string; status: 'completed' | 'failed' | 'processing'; result: any; error: string | null }) => void) => () => void;
}

export function createMockTaskRunner(): MockTaskRunner {
  const callbacks: Array<(result: any) => void> = [];
  const timers: ReturnType<typeof setTimeout>[] = [];

  return {
    start(sequenceBlockId: string) {
      const jobId = `mock_job_${Date.now()}`;

      // Simulate queued -> processing -> completed after delays
      const t1 = setTimeout(() => {
        const t2 = setTimeout(() => {
          const mockResult = {
            classification: MOCK_CLASSIFICATION,
            attention: MOCK_ATTENTION(SAMPLE_SEQUENCE),
            gcn: MOCK_GCN(SAMPLE_SEQUENCE),
          };
          callbacks.forEach((cb) =>
            cb({
              sequenceBlockId,
              status: 'completed',
              result: mockResult,
              error: null,
            })
          );
        }, 1500); // processing -> completed after 1.5s
        timers.push(t2);

        callbacks.forEach((cb) =>
          cb({
            sequenceBlockId,
            status: 'processing',
            result: null,
            error: null,
          })
        );
      }, 800); // queued -> processing after 0.8s
      timers.push(t1);

      return jobId;
    },
    onResult(callback: (result: any) => void) {
      callbacks.push(callback);
      return () => {
        const idx = callbacks.indexOf(callback);
        if (idx > -1) callbacks.splice(idx, 1);
      };
    },
  };
}

// ==================== Create Default Sequence Block ====================

export function createDefaultSequenceBlock(dataset?: DatasetType): SequenceBlock {
  return {
    id: generateId('seq'),
    type: 'sequence',
    title: dataset || 'Human',
    dataset: dataset || 'Human',
    sequenceCount: 1000,
    sequence: '',
    status: 'idle',
    jobId: null,
    resultSummary: null,
    boundModelId: null,
  };
}

// ==================== Create Visualization Block ====================

export function createVizBlock(
  vizType: VisualizationBlock['vizType'],
  title?: string
): VisualizationBlock {
  const typeInfo = {
    classification: 'Classification',
    attention: 'Attention',
    'attention-score': 'Attention Score',
    'gcn-graph': 'GCN Graph',
    'gcn-message-passing': 'GCN Message Passing',
    'integrated-gradients': 'Integrated Gradients',
    'model-graph': 'Model Graph',
  };

  return {
    id: generateId('viz'),
    type: 'visualization',
    title: title || typeInfo[vizType] || 'Visualization',
    vizType,
    status: 'idle',
    boundSequenceId: null,
    boundModelId: null,
    params: getDefaultParams(vizType),
    result: null,
    autoRun: false,
  };
}

function getDefaultParams(vizType: VisualizationBlock['vizType']): Record<string, any> {
  switch (vizType) {
    case 'attention':
      return { topX: 3, modificationType: 'all' };
    case 'integrated-gradients':
      return { targetClassId: 0, topN: 10 };
    case 'gcn-message-passing':
      return { targetNodeIdx: 0 };
    default:
      return {};
  }
}

// ==================== Utilities ====================

export function countChars(seq: string): Record<string, number> {
  const counts: Record<string, number> = { A: 0, C: 0, G: 0, T: 0, U: 0, N: 0 };
  for (const ch of seq) {
    const upper = ch.toUpperCase();
    if (upper in counts) {
      counts[upper]++;
    }
  }
  return counts;
}

export function validateSequence(seq: string): { isValid: boolean; errors: string[] } {
  const errors: string[] = [];
  if (seq.length === 0) {
    errors.push('Sequence is empty.');
  }
  if (seq.length > 1001) {
    errors.push('Sequence exceeds maximum length of 1001.');
  }
  const invalidChars = seq.replace(/[ACGTUN]/gi, '');
  if (invalidChars.length > 0) {
    errors.push(`Invalid characters found: ${[...new Set(invalidChars.split(''))].join(', ')}`);
  }
  return { isValid: errors.length === 0, errors };
}