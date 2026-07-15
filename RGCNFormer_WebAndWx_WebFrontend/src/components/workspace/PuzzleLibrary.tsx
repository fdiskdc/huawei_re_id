/**
 * PuzzleLibrary.tsx - 可拖入 Workspace 的"积木"库(左栏)/ Puzzle library (left sidebar)
 *
 * WorkspacePage 左侧栏,提供两个分组:
 *   1. Input Blocks:Human / Plant / 3Gen 三个数据集选项(点击 → onAddSequence)
 *   2. Visualization Blocks:VIZ_TYPE_REGISTRY 中所有可视化类型(点击 → onAddViz)
 * 每项展示图标、名称、简短描述。点击调用父组件回调,把对应 block 添加到 Workspace 状态。
 * Left sidebar of WorkspacePage providing two groups of draggable blocks:
 *   1. Input Blocks: Human/Plant/3Gen dataset options (click → onAddSequence)
 *   2. Visualization Blocks: all viz types from VIZ_TYPE_REGISTRY (click → onAddViz)
 * Each item shows an icon, name, and short description. Clicking invokes a parent callback
 * that appends the corresponding block to the workspace state.
 *
 * 功能模块 / Modules:
 * - DATASET_ICONS: DatasetType → emoji 图标 / Dataset → emoji icon
 * - Input Blocks 列表(DATASET_OPTIONS 渲染)/ Input blocks list
 * - Visualization Blocks 列表(VIZ_TYPE_REGISTRY 渲染)/ Viz blocks list
 * - onAddSequence / onAddViz 回调透传 / Pass-through click callbacks
 *
 * 输入 / Inputs:
 * - onAddSequence: (dataset: DatasetType) => void
 * - onAddViz: (vizType: VizType) => void
 *
 * 输出 / Outputs:
 * - JSX.Element 列表 UI / List UI
 *
 * 数据流 / Data Flow:
 * 1. WorkspacePage 渲染 <PuzzleLibrary onAddSequence onAddViz />
 * 2. 用户点击 Input/Visualization 项
 * 3. WorkspacePage 收到回调 → 创建 block → 加入 sequenceBlocks/vizBlocks
 * 4. WorkspaceCanvas 重渲染,显示新 block 卡片
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: workspace/types(VIZ_TYPE_REGISTRY, DATASET_OPTIONS, DATASET_LABELS)
 * - 被调用 / Called by: pages/WorkspacePage.tsx
 * - 关联 / Related: WorkspaceCanvas 展示新 block
 *
 * 使用示例 / Usage Example:
 *   <PuzzleLibrary
 *     onAddSequence={(ds) => addSequenceBlock(ds)}
 *     onAddViz={(vt) => addVizBlock(vt)}
 *   />
 */

import React from 'react';
import { VIZ_TYPE_REGISTRY, DATASET_OPTIONS, DATASET_LABELS } from './types';
import type { VizType, DatasetType } from './types';

interface PuzzleLibraryProps {
  onAddSequence: (dataset: DatasetType) => void;
  onAddViz: (vizType: VizType) => void;
}

const DATASET_ICONS: Record<DatasetType, string> = {
  Human: '🧬',
  Plant: '🌱',
  '3Gen': '🧬',
};

const PuzzleLibrary: React.FC<PuzzleLibraryProps> = ({ onAddSequence, onAddViz }) => {
  return (
    <div className="puzzle-library">
      <h3>Input Blocks</h3>
      {DATASET_OPTIONS.map((dataset) => (
        <div
          key={dataset}
          className="puzzle-item"
          onClick={() => onAddSequence(dataset)}
        >
          <div className="puzzle-item-icon input">{DATASET_ICONS[dataset]}</div>
          <div className="puzzle-item-text">
            <span className="puzzle-item-name">{DATASET_LABELS[dataset]}</span>
            <span className="puzzle-item-desc">RNA sequence — {dataset} dataset</span>
          </div>
        </div>
      ))}

      <h3>Visualization Blocks</h3>
      {VIZ_TYPE_REGISTRY.map((viz) => (
        <div
          key={viz.key}
          className="puzzle-item"
          onClick={() => onAddViz(viz.key)}
        >
          <div className="puzzle-item-icon viz">{viz.icon}</div>
          <div className="puzzle-item-text">
            <span className="puzzle-item-name">{viz.label}</span>
            <span className="puzzle-item-desc">{viz.description.slice(0, 30)}...</span>
          </div>
        </div>
      ))}
    </div>
  );
};

export default PuzzleLibrary;
