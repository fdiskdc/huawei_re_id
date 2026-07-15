/**
 * VizBlockCard.tsx - 可视化块卡片(绑模型 + 数据源)/ Visualization block card
 *
 * Workspace 工作区"Visualization Zone"中的卡片,代表用户从 PuzzleLibrary 拖入的
 * 一个可视化模块(attention / IG / GCN 等)。展示 viz 类型(从 VIZ_TYPE_REGISTRY
 * 查 icon 与 label)、当前绑定的模型("Model: ..." 或 "No model bound")、
 * 绑定的数据源标题(若有),以及状态(idle / ready / running / completed / failed)。
 * 卡片颜色由绑定模型 id 决定 — 同一模型在 Input/Model/Visualization 三个区共享颜色。
 * Card inside the workspace's "Visualization Zone" rendering a viz block
 * (attention/IG/GCN/etc.) added from PuzzleLibrary. Shows viz type (icon/label from
 * VIZ_TYPE_REGISTRY), bound model, bound data source label, and status. Card color is
 * derived from the bound model id — the same model shares a color across the three zones.
 *
 * 功能模块 / Modules:
 * - viz 元信息查表(VIZ_TYPE_REGISTRY)/ Viz metadata lookup
 * - 绑定模型查找(modelBlocks.find)/ Bound-model lookup
 * - 状态标签映射(5 状态)/ Status label map (5 states)
 * - autoRun 标志(可视化卡额外属性)/ autoRun indicator
 *
 * 输入 / Inputs:
 * - block: VisualizationBlock / viz block
 * - isSelected: boolean / selection flag
 * - onClick: () => void / click callback
 * - modelBlocks: ModelBlock[] / all model blocks in workspace
 * - boundSequenceLabel?: string / title of the bound sequence (optional)
 *
 * 输出 / Outputs:
 * - JSX.Element 卡片 / Card JSX
 *
 * 数据流 / Data Flow:
 * 1. 用户在 PuzzleLibrary 选中 visualization 类型 → Workspace 添加 vizBlock
 * 2. 用户在 PropertiesPanel 绑定数据源(sequenceBlock)与模型
 * 3. 点击"Apply" → onRunViz → 后端推理 → 状态变为 running → completed
 * 4. 卡片刷新显示新状态
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: workspace/types(VIZ_TYPE_REGISTRY, BINDING_COLORS, getBindingColorIndex)
 * - 被调用 / Called by: components/workspace/WorkspaceCanvas.tsx
 * - 关联 / Related: SequenceBlockCard, ModelBlockCard(共用 BINDING_COLORS)
 *
 * 使用示例 / Usage Example:
 *   <VizBlockCard
 *     block={vizBlock}
 *     isSelected={selectedId === vizBlock.id}
 *     onClick={() => onSelectBlock(vizBlock.id)}
 *     modelBlocks={modelBlocks}
 *     boundSequenceLabel="Human 1000 (Human)"
 *   />
 */

import React from 'react';
import type { VisualizationBlock, ModelBlock } from './types';
import { VIZ_TYPE_REGISTRY, BINDING_COLORS, getBindingColorIndex } from './types';

interface VizBlockCardProps {
  block: VisualizationBlock;
  isSelected: boolean;
  onClick: () => void;
  modelBlocks: ModelBlock[];
  boundSequenceLabel?: string;
}

const STATUS_LABELS: Record<string, string> = {
  idle: 'Idle',
  ready: 'Ready',
  running: 'Running...',
  completed: 'Completed',
  failed: 'Failed',
};

const VizBlockCard: React.FC<VizBlockCardProps> = ({
  block,
  isSelected,
  onClick,
  modelBlocks,
  boundSequenceLabel,
}) => {
  const meta = VIZ_TYPE_REGISTRY.find((v) => v.key === block.vizType);
  const boundModel = modelBlocks.find((m) => m.id === block.boundModelId);

  // Binding color based on model (same model = same color)
  const colorIdx = getBindingColorIndex(block.boundModelId);
  const colors = colorIdx >= 0 ? BINDING_COLORS[colorIdx] : null;

  const cardStyle: React.CSSProperties = colors
    ? {
        borderColor: isSelected ? colors.accent : colors.border,
        background: colors.bg,
        ...(isSelected ? { boxShadow: `0 0 0 2px ${colors.accent}33` } : {}),
      }
    : {};

  return (
    <div
      className={`block-card ${isSelected ? 'selected' : ''}`}
      onClick={onClick}
      style={cardStyle}
    >
      <div className="block-card-title">
        {meta?.icon || '📊'} {block.title}
      </div>
      <div className="block-card-subtitle">
        {meta?.label || block.vizType}
      </div>
      <div className="viz-card-binding">
        {boundModel ? (
          <span style={colors ? { color: colors.accent } : undefined}>
            Model: {boundModel.title}
          </span>
        ) : (
          <span style={{ color: 'var(--ws-text-muted)', fontStyle: 'italic' }}>
            No model bound
          </span>
        )}
      </div>
      {boundSequenceLabel && (
        <div className="block-card-meta" style={{ marginTop: 2 }}>
          <span style={{ fontSize: 10, color: 'var(--ws-text-muted)' }}>
            Data: {boundSequenceLabel}
          </span>
        </div>
      )}
      <div className="block-card-meta" style={{ marginTop: 6 }}>
        <span className={`status-badge ${block.status}`}>
          {STATUS_LABELS[block.status] || block.status}
        </span>
        {block.autoRun && (
          <span style={{ marginLeft: 4, fontSize: 10, color: 'var(--ws-text-muted)' }}>
            Auto-run
          </span>
        )}
      </div>
    </div>
  );
};

export default VizBlockCard;
