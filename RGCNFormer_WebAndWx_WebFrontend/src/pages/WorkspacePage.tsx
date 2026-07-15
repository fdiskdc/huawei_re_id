/**
 * WorkspacePage.tsx - 积木式工作台(单页可视化分析)/ Puzzle-style analysis workbench
 *
 * / 路由页面(根页面,替代旧 MainPage)。整个页面分为三栏:
 *   - 左:PuzzleLibrary(可拖入 Input/Model/Visualization block)
 *   - 中:WorkspaceCanvas(三区布局:Input/Model/Visualization)
 *   - 右:PropertiesPanel(选中 block 的属性 + 编辑)
 * 状态全部由 WorkspacePage 顶层 useState 维护(sequenceBlocks / modelBlocks / vizBlocks /
 * selectedBlockId),子组件通过 props 接收。
 * Page mounted at / (root, replacing the legacy MainPage). Three-column layout:
 *   - Left: PuzzleLibrary (Input/Model/Visualization block source)
 *   - Center: WorkspaceCanvas (three-zone layout: Input/Model/Visualization)
 *   - Right: PropertiesPanel (selected block's properties + editor)
 * All state lives in WorkspacePage via useState (sequenceBlocks/modelBlocks/vizBlocks/
 * selectedBlockId); child components receive props only.
 *
 * 功能模块 / Modules:
 * - 三栏布局(Grid/Flex)/ Three-column layout
 * - sequenceBlocks/modelBlocks/vizBlocks 状态 / Workspace state
 * - selectedBlockId 选中态 / Selection state
 * - onAddSequence / onAddViz / onUpdateBlock / onRemoveBlock / onRunViz 回调
 *   / Add/Update/Remove/Run callbacks
 * - Apply 后跳 VizDisplayPage / Navigate to VizDisplayPage after run
 *
 * 输入 / Inputs:
 * - 用户点击 PuzzleLibrary 项(添加 block)
 * - 用户在 PropertiesPanel 编辑(更新 block)
 * - 用户点击 Apply(运行 viz)→ 调后端
 *
 * 输出 / Outputs:
 * - JSX.Element 三栏工作台 / Three-column workbench JSX
 *
 * 数据流 / Data Flow:
 * 1. 初始:sequenceBlocks=[], modelBlocks=MODEL_REGISTRY, vizBlocks=[]
 * 2. 用户在 PuzzleLibrary 选 dataset → onAddSequence → 推入 sequenceBlocks
 * 3. 用户在 PropertiesPanel 改序列/绑模型 → onUpdateBlock
 * 4. 用户点击 Apply → onRunViz → fetch + setResult → 跳转 VizDisplayPage
 * 5. selectedBlockId 变化 → 右侧 PropertiesPanel 重渲染
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: 所有 workspace/* 子组件、context/RnaContext、lib/api
 * - 被调用 / Called by: App.tsx(<Route path="/">)
 * - 关联 / Related: VizDisplayPage(展示完整结果)
 *
 * 使用示例 / Usage Example:
 *   <Route path="/" element={<WorkspacePage />} />
 *   // 浏览器 /mrmodn/
 */

import React, { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Modal, message } from 'antd';
import './WorkspacePage.css';
import { useRna } from '../context/RnaContext';
import PuzzleLibrary from '../components/workspace/PuzzleLibrary';
import WorkspaceCanvas from '../components/workspace/WorkspaceCanvas';
import PropertiesPanel from '../components/workspace/PropertiesPanel';
import type {
  SequenceBlock,
  ModelBlock,
  VisualizationBlock,
  WorkspaceBlock,
  VizType,
  DatasetType,
} from '../components/workspace/types';
import { VIZ_TYPE_REGISTRY } from '../components/workspace/types';
import {
  DEFAULT_MODELS,
  createDefaultSequenceBlock,
  createVizBlock,
} from '../components/workspace/mockData';
import { submitTask, fetchResult, isCompleted, isFailed, generateJobId } from '../lib/api';

// Re-use existing visualization components for rendering in the viz zone
import ClassificationViz from './ClassificationViz';
import AttentionViz from './AttentionViz';
import GcnViz from './GcnViz';
import TargetGcnViz from './TargetGcnViz';
import IntegratedGradientsViz from './IntegratedGradientsViz';
import ModelViz from './ModelViz';
import AttentionDistributionViz from './AttentionDistributionViz';

const WorkspacePage: React.FC = () => {
  const navigate = useNavigate();
  const { setRnaSequence } = useRna();

  // ==================== State ====================
  const [sequenceBlocks, setSequenceBlocks] = useState<SequenceBlock[]>([]);
  const [modelBlocks] = useState<ModelBlock[]>(DEFAULT_MODELS);
  const [vizBlocks, setVizBlocks] = useState<VisualizationBlock[]>([]);
  const [selectedBlockId, setSelectedBlockId] = useState<string | null>(null);
  const [vizModalVisible, setVizModalVisible] = useState(false);
  const [vizModalBlock, setVizModalBlock] = useState<VisualizationBlock | null>(null);

  // Sync RNA sequence to context whenever sequence blocks change
  useEffect(() => {
    const firstSeqWithSequence = sequenceBlocks.find((b) => b.sequence);
    if (firstSeqWithSequence) {
      setRnaSequence(firstSeqWithSequence.sequence);
    }
  }, [sequenceBlocks, setRnaSequence]);

  // ==================== Block Operations ====================

  const addSequenceBlock = useCallback(
    (dataset: DatasetType) => {
      const newBlock = createDefaultSequenceBlock(dataset);
      const count = sequenceBlocks.filter((b) => b.dataset === dataset).length + 1;
      newBlock.title = `${dataset}-${String(count).padStart(3, '0')}`;
      setSequenceBlocks((prev) => [...prev, newBlock]);
      setSelectedBlockId(newBlock.id);
    },
    [sequenceBlocks]
  );

  const addVizBlock = useCallback(
    (vizType: VizType) => {
      const newBlock = createVizBlock(vizType);
      setVizBlocks((prev) => [...prev, newBlock]);
      setSelectedBlockId(newBlock.id);
    },
    []
  );

  const removeBlock = useCallback((id: string) => {
    setSequenceBlocks((prev) => prev.filter((b) => b.id !== id));
    setVizBlocks((prev) => prev.filter((b) => b.id !== id));
    setSelectedBlockId((prev) => (prev === id ? null : prev));
  }, []);

  const updateBlock = useCallback(
    (id: string, updates: Partial<WorkspaceBlock>) => {
      setSequenceBlocks((prev) =>
        prev.map((b) => (b.id === id ? { ...b, ...updates } as SequenceBlock : b))
      );
      setVizBlocks((prev) =>
        prev.map((b) => {
          if (b.id !== id) return b;
          let merged = { ...b, ...updates } as VisualizationBlock;
          // Auto-derive boundModelId from bound sequence when sequence binding changes
          if ('boundSequenceId' in updates && !('boundModelId' in updates)) {
            const seqBlock = sequenceBlocks.find((s) => s.id === merged.boundSequenceId);
            merged = { ...merged, boundModelId: seqBlock?.boundModelId || null };
          }
          return merged;
        })
      );
    },
    [sequenceBlocks]
  );

  // ==================== Run Visualization ====================

  const runVisualization = useCallback(
    async (vizBlockId: string) => {
      const vizBlock = vizBlocks.find((b) => b.id === vizBlockId);
      if (!vizBlock || !vizBlock.boundSequenceId) return;

      const seqBlock = sequenceBlocks.find((s) => s.id === vizBlock.boundSequenceId);
      if (!seqBlock || !seqBlock.sequence) return;

      // Check if the bound model is not yet implemented
      const IMPLEMENTED_MODELS = ['model_mrmodn'];
      if (vizBlock.boundModelId && !IMPLEMENTED_MODELS.includes(vizBlock.boundModelId)) {
        message.warning('Model Coming Soon — This model is not yet available.');
        return;
      }

      // Set viz block to running
      setVizBlocks((prev) =>
        prev.map((b) =>
          b.id === vizBlockId ? { ...b, status: 'running' as const } : b
        )
      );

      try {
        // Submit task to backend
        const submitResponse = await submitTask({
          jobId: generateJobId(),
          userId: 'user1',
          rnaSequence: seqBlock.sequence,
        });

        let resultData: any = null;

        // Check if result was returned directly (cached)
        if (isCompleted(submitResponse as any)) {
          resultData = submitResponse;
        } else {
          // Poll for result
          const jobId = submitResponse.jobId;
          const maxAttempts = 100;
          let attempts = 0;

          while (attempts < maxAttempts) {
            await new Promise((resolve) => setTimeout(resolve, 3000));
            const result = await fetchResult(jobId);

            if (isCompleted(result)) {
              resultData = result;
              break;
            }
            if (isFailed(result)) {
              throw new Error(result.error || 'Task failed');
            }
            attempts++;
          }

          if (!resultData) {
            throw new Error('Timeout: task did not complete within expected time');
          }
        }

        // Update viz block with result
        setVizBlocks((prev) => {
          const updated = prev.map((b) =>
            b.id === vizBlockId
              ? { ...b, status: 'completed' as const, result: resultData }
              : b
          );
          const completed = updated.find((b) => b.id === vizBlockId);
          if (completed && completed.result) {
            setVizModalBlock(completed);
            setVizModalVisible(true);
          }
          return updated;
        });
      } catch (error: any) {
        setVizBlocks((prev) =>
          prev.map((b) =>
            b.id === vizBlockId ? { ...b, status: 'failed' as const } : b
          )
        );
      }
    },
    [vizBlocks, sequenceBlocks]
  );

  const handleCloseVizModal = useCallback(() => {
    setVizModalVisible(false);
    setVizModalBlock(null);
  }, []);

  // ==================== Derived State ====================

  const selectedBlock: WorkspaceBlock | null = (() => {
    if (!selectedBlockId) return null;
    return (
      sequenceBlocks.find((b) => b.id === selectedBlockId) ||
      modelBlocks.find((b) => b.id === selectedBlockId) ||
      vizBlocks.find((b) => b.id === selectedBlockId) ||
      null
    );
  })();

  return (
    <div className="workspace-root">
      {/* Header */}
      <div className="workspace-header">
        <h1>🧬 mRNA Modification Analysis Workbench</h1>
        <div className="workspace-header-actions">
          <span style={{ fontSize: 12, color: 'var(--ws-text-muted)' }}>
            {sequenceBlocks.length} sequences · {vizBlocks.length} visualizations
          </span>
          <button
            className="process-btn"
            onClick={() => navigate('/nextgen/compare')}
            title="Compare model performance"
          >
            Compare Page
          </button>
        </div>
      </div>

      {/* Body: Library | Canvas | Properties */}
      <div className="workspace-body">
        <PuzzleLibrary onAddSequence={addSequenceBlock} onAddViz={addVizBlock} />

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <WorkspaceCanvas
            sequenceBlocks={sequenceBlocks}
            modelBlocks={modelBlocks}
            vizBlocks={vizBlocks}
            selectedBlockId={selectedBlockId}
            onSelectBlock={setSelectedBlockId}
          />
        </div>

        <PropertiesPanel
          block={selectedBlock}
          sequenceBlocks={sequenceBlocks}
          modelBlocks={modelBlocks}
          onUpdateBlock={updateBlock}
          onRemoveBlock={removeBlock}
          onRunViz={runVisualization}
        />
      </div>

      {/* Visualization Result Modal */}
      <Modal
        title={
          vizModalBlock
            ? `${VIZ_TYPE_REGISTRY.find((v) => v.key === vizModalBlock.vizType)?.icon || '📊'} ${
                vizModalBlock.title || vizModalBlock.vizType
              } - 可视化结果`
            : '可视化结果'
        }
        open={vizModalVisible}
        onCancel={handleCloseVizModal}
        footer={null}
        width={900}
        destroyOnClose
        styles={{ body: { padding: 16, height: '85vh', overflow: 'auto' } }}
      >
        {vizModalBlock && <VizRenderer vizBlock={vizModalBlock} />}
      </Modal>
    </div>
  );
};

// ==================== VizRenderer ====================

interface VizRendererProps {
  vizBlock: VisualizationBlock;
}

const VizRenderer: React.FC<VizRendererProps> = ({ vizBlock }) => {
  if (!vizBlock.result) {
    return (
      <div className="viz-result-placeholder">
        <div className="placeholder-icon">📊</div>
        <div>No results available</div>
      </div>
    );
  }

  const vizType = vizBlock.vizType;

  // Render based on visualization type
  try {
    switch (vizType) {
      case 'classification':
        return (
          <div style={{ border: '1px solid var(--ws-border)', borderRadius: 8, padding: 12, background: 'white' }}>
            <ClassificationViz data={vizBlock.result.classification} />
          </div>
        );
      case 'attention':
        return (
          <div style={{ border: '1px solid var(--ws-border)', borderRadius: 8, padding: 12, background: 'white' }}>
            <AttentionViz data={vizBlock.result.attention} />
          </div>
        );
      case 'gcn-graph':
        return (
          <div style={{ border: '1px solid var(--ws-border)', borderRadius: 8, padding: 12, background: 'white', height: '70vh', minHeight: 400 }}>
            <GcnViz data={vizBlock.result.gcn} />
          </div>
        );
      case 'gcn-message-passing':
        return (
          <div style={{ border: '1px solid var(--ws-border)', borderRadius: 8, padding: 12, background: 'white', minHeight: 500 }}>
            <TargetGcnViz targetNodeIdx={vizBlock.params.targetNodeIdx || 0} data={vizBlock.result.gcnAggregation} />
          </div>
        );
      case 'integrated-gradients':
        return (
          <div style={{ border: '1px solid var(--ws-border)', borderRadius: 8, padding: 12, background: 'white', minHeight: 500 }}>
            <IntegratedGradientsViz data={vizBlock.result.integratedGradients} />
          </div>
        );
      case 'model-graph':
        return (
          <div style={{ border: '1px solid var(--ws-border)', borderRadius: 8, padding: 12, background: 'white' }}>
            <ModelViz />
          </div>
        );
      case 'attention-score':
        return (
          <div style={{ border: '1px solid var(--ws-border)', borderRadius: 8, padding: 12, background: 'white' }}>
            <AttentionDistributionViz />
          </div>
        );
      default:
        return (
          <div className="viz-result-placeholder">
            <div className="placeholder-icon">❓</div>
            <div>Unknown visualization type: {vizType}</div>
          </div>
        );
    }
  } catch (e: any) {
    return (
      <div style={{ padding: 16, background: '#f0dede', borderRadius: 8 }}>
        <strong>Error rendering {vizType}:</strong> {e.message}
      </div>
    );
  }
};

export default WorkspacePage;
