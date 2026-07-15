/**
 * VizDisplayPage.tsx - 全屏可视化展示页(由 Workspace 注入完成数据)/ Full-screen viz display
 *
 * /viz-display 路由页面。接收 WorkspacePage 通过 useNavigate state 传入的
 * 已完成 viz 数据(vizBlocks + sequenceBlocks),用全屏布局渲染:左侧 viz 块
 * 列表(点击切换)、右侧对应 viz 子组件。
 * Page mounted at /viz-display. Receives completed viz data (vizBlocks + sequenceBlocks)
 * from WorkspacePage via useNavigate state, and renders a full-screen layout:
 * left = viz block list (click to switch), right = corresponding viz sub-component.
 *
 * 功能模块 / Modules:
 * - useLocation().state 读取 Workspace 传入的数据
 *   / Read state from useLocation
 * - 左栏 viz 列表 / Left viz list
 * - 右栏对应 viz 组件 / Right viz renderer
 * - 返回 Workspace 按钮 / Return to Workspace button
 *
 * 输入 / Inputs:
 * - location.state: { vizBlocks, sequenceBlocks, modelBlocks, ... }
 *   (由 WorkspacePage.navigate('/viz-display', { state }) 注入)
 *   / State injected by WorkspacePage
 *
 * 输出 / Outputs:
 * - JSX.Element 全屏 viz 展示页 / Full-screen viz display JSX
 *
 * 数据流 / Data Flow:
 * 1. WorkspacePage Apply 完成 → navigate('/viz-display', { state })
 * 2. VizDisplayPage 挂载 → useLocation().state 拿数据
 * 3. 渲染左栏 viz 列表
 * 4. 默认选中第一个 viz,右栏渲染
 * 5. 用户切换左栏 → 右栏重渲染
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: context/RnaContext、workspace/types、所有 viz 子组件
 * - 被调用 / Called by: App.tsx(<Route path="/viz-display">)
 * - 关联 / Related: WorkspacePage(数据源)、ResultsPage(URL jobId 版)
 *
 * 使用示例 / Usage Example:
 *   // WorkspacePage.tsx 内
 *   navigate('/viz-display', { state: { vizBlocks, sequenceBlocks, modelBlocks } });
 *   // 浏览器 /mrmodn/viz-display
 */

import React, { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import type { SequenceBlock, VisualizationBlock } from '../components/workspace/types';
import { useRna, type ClassificationResult } from '../context/RnaContext';

import ClassificationViz from './ClassificationViz';
import AttentionViz from './AttentionViz';
import AttentionDistributionViz from './AttentionDistributionViz';
import GcnViz from './GcnViz';
import TargetGcnViz from './TargetGcnViz';
import IntegratedGradientsViz from './IntegratedGradientsViz';
import ModelViz from './ModelViz';

interface VizDisplayState {
  sequenceBlocks: SequenceBlock[];
  vizBlocks: VisualizationBlock[];
}

// Helper to extract leaf classification results from tree structure
function extractClassificationResults(tree: any): ClassificationResult[] {
  const results: ClassificationResult[] = [];

  function traverse(node: any) {
    if (!node) return;

    // If this is a leaf node (no children) and has isPredicted
    if (!node.children || node.children.length === 0) {
      results.push({
        name: node.name,
        value: node.isPredicted ? 1 : 0,
      });
    } else {
      // Recursively traverse children
      for (const child of node.children) {
        traverse(child);
      }
    }
  }

  traverse(tree);
  return results;
}

const VizDisplayPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const state = location.state as VizDisplayState | null;
  const { setRnaSequence, setClassificationResults } = useRna();

  // Set sequence in RnaContext from sequenceBlocks
  useEffect(() => {
    if (state?.sequenceBlocks && state.sequenceBlocks.length > 0) {
      const firstSeqBlock = state.sequenceBlocks[0];
      if (firstSeqBlock.sequence) {
        setRnaSequence(firstSeqBlock.sequence);
      }
      if (firstSeqBlock.resultSummary?.classification) {
        // Extract classification results from tree structure
        const results = extractClassificationResults(firstSeqBlock.resultSummary.classification);
        setClassificationResults(results);
      }
    }
  }, [state, setRnaSequence, setClassificationResults]);

  if (!state || !state.vizBlocks || state.vizBlocks.length === 0) {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: '#f5f3ef',
        color: '#7d7872',
        gap: 16,
      }}>
        <div style={{ fontSize: 48, opacity: 0.4 }}>📊</div>
        <p>No visualization data available.</p>
        <button
          onClick={() => navigate('/nextgen')}
          style={{
            padding: '8px 24px',
            border: 'none',
            borderRadius: 6,
            background: '#8a9eb5',
            color: 'white',
            fontSize: 14,
            cursor: 'pointer',
          }}
        >
          Back to Workspace
        </button>
      </div>
    );
  }

  const { sequenceBlocks, vizBlocks } = state;

  const handleBack = () => {
    navigate('/nextgen');
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: '#f5f3ef',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
        height: 56,
        background: '#ffffff',
        borderBottom: '1px solid #ddd8d2',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button
            onClick={handleBack}
            style={{
              padding: '6px 16px',
              border: '1px solid #ddd8d2',
              borderRadius: 6,
              background: '#f5f3ef',
              color: '#3a3632',
              fontSize: 13,
              cursor: 'pointer',
            }}
          >
            ← Back
          </button>
          <h1 style={{
            fontSize: 16,
            fontWeight: 600,
            color: '#3a3632',
            margin: 0,
          }}>
            📊 Visualization Display
          </h1>
        </div>
        <div style={{ fontSize: 12, color: '#a8a29e' }}>
          {vizBlocks.length} visualization{vizBlocks.length > 1 ? 's' : ''} · {sequenceBlocks.length} sequence source{vizBlocks.length > 1 ? 's' : ''}
        </div>
      </div>

      {/* Content */}
      <div style={{
        flex: 1,
        padding: 24,
        overflow: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: 24,
      }}>
        {vizBlocks.map((vizBlock) => {
          if (!vizBlock.result) return null;

          const seqBlock = sequenceBlocks.find((s) => s.id === vizBlock.boundSequenceId);
          const vizTitle = vizBlock.title || vizBlock.vizType;

          return (
            <div
              key={vizBlock.id}
              style={{
                background: '#ffffff',
                border: '1px solid #ddd8d2',
                borderRadius: 10,
                overflow: 'hidden',
              }}
            >
              {/* Viz Block Header */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '12px 16px',
                background: '#faf9f7',
                borderBottom: '1px solid #ece8e3',
              }}>
                <div style={{ fontWeight: 600, fontSize: 14, color: '#3a3632' }}>
                  {vizTitle}
                </div>
                <div style={{ fontSize: 11, color: '#a8a29e' }}>
                  {seqBlock ? `Source: ${seqBlock.title} (${seqBlock.dataset})` : 'No source'}
                </div>
              </div>

              {/* Viz Block Content */}
              <div style={{ padding: 16 }}>
                <VizDisplayContent vizBlock={vizBlock} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ==================== Viz Display Content ====================

interface VizDisplayContentProps {
  vizBlock: VisualizationBlock;
}

const VizDisplayContent: React.FC<VizDisplayContentProps> = ({ vizBlock }) => {
  const vizType = vizBlock.vizType;

  // attention-score fetches its own data via useRna(), doesn't need vizBlock.result
  if (vizType === 'attention-score') {
    return <AttentionDistributionViz />;
  }

  if (!vizBlock.result) {
    return <div style={{ color: '#a8a29e', textAlign: 'center', padding: 40 }}>No result data</div>;
  }

  try {
    switch (vizType) {
      case 'classification':
        return <ClassificationViz data={vizBlock.result.classification} />;
      case 'attention':
        return <AttentionViz data={vizBlock.result.attention} />;
      case 'gcn-graph':
        return <GcnViz data={vizBlock.result.gcn} />;
      case 'gcn-message-passing':
        return (
          <TargetGcnViz targetNodeIdx={vizBlock.params.targetNodeIdx || 0} />
        );
      case 'integrated-gradients':
        return <IntegratedGradientsViz />;
      case 'model-graph':
        return <ModelViz />;
      default:
        return (
          <div style={{ color: '#a8a29e', textAlign: 'center', padding: 40 }}>
            Unknown visualization type: {vizType}
          </div>
        );
    }
  } catch (e: any) {
    return (
      <div style={{ padding: 16, background: '#f0dede', borderRadius: 8, color: '#c48a8a' }}>
        <strong>Error rendering {vizType}:</strong> {e.message}
      </div>
    );
  }
};

export default VizDisplayPage;
