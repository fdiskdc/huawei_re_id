/**
 * SequenceBlockCard.tsx - Input Zone 数据集卡片 / Input Zone dataset card
 *
 * Workspace 工作区中"Input Zone"里的卡片,代表用户从 PuzzleLibrary 拖入的
 * 一个 RNA 序列数据集块(Human / Plant / 3Gen 等)。展示标题、数据集标签、
 * 序列数量、当前状态(idle/validating/.../failed),以及绑定的模型(若有)。
 * 卡片颜色由 BINDING_COLORS 与绑定模型 id 决定;选中态有高亮 box-shadow。
 * Card used inside the workspace's "Input Zone" to render a dataset block (Human/Plant/3Gen)
 * dropped from PuzzleLibrary. Renders title, dataset tag, sequence count, lifecycle status
 * (idle/validating/.../failed), and bound model (if any). Card colors are derived from
 * BINDING_COLORS and the bound model id; the selected state adds a highlight box-shadow.
 *
 * 功能模块 / Modules:
 * - STATUS_LABELS: 8 状态英文文案 / 8 status → English label map
 * - 绑定色计算 (getBindingColorIndex / BINDING_COLORS) / Binding color helpers
 * - 卡片渲染:标题/子标题/状态徽章/绑定模型提示 / Card render
 *
 * 输入 / Inputs:
 * - block: SequenceBlock(workspace/types.ts)/ Sequence block (workspace/types.ts)
 * - isSelected: boolean / selection flag
 * - onClick: () => void / click callback
 *
 * 输出 / Outputs:
 * - JSX.Element 卡片 / Card JSX
 *
 * 数据流 / Data Flow:
 * 1. WorkspaceCanvas 把 sequenceBlocks 列表传进来
 * 2. 用户点击卡片 → onSelectBlock(block.id)
 * 3. PropertiesPanel 展示该 SequenceBlock 的可编辑属性
 * 4. 状态变化(idle→submitted→processing→completed)由父组件驱动重渲染
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: workspace/types(getBindingColorIndex, BINDING_COLORS)
 * - 被调用 / Called by: components/workspace/WorkspaceCanvas.tsx
 * - 数据来源 / Data: WorkspacePage.tsx(state)、lib/api.ts(后端任务状态轮询)
 *
 * 使用示例 / Usage Example:
 *   <SequenceBlockCard
 *     block={seqBlock}
 *     isSelected={selectedId === seqBlock.id}
 *     onClick={() => onSelectBlock(seqBlock.id)}
 *   />
 */

import React from 'react';
import type { SequenceBlock } from './types';
import { BINDING_COLORS, getBindingColorIndex } from './types';

interface SequenceBlockCardProps {
  block: SequenceBlock;
  isSelected: boolean;
  onClick: () => void;
}

const STATUS_LABELS: Record<string, string> = {
  idle: 'Idle',
  validating: 'Validating...',
  validated: 'Validated',
  submitting: 'Submitting...',
  submitted: 'Submitted',
  processing: 'Processing...',
  completed: 'Completed',
  failed: 'Failed',
};

const SequenceBlockCard: React.FC<SequenceBlockCardProps> = ({
  block,
  isSelected,
  onClick,
}) => {
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
        🧬 {block.title}
        <span className="dataset-tag">{block.dataset}</span>
      </div>
      <div className="block-card-subtitle">
        {block.sequenceCount.toLocaleString()} sequences
      </div>
      <div className="block-card-meta" style={{ marginTop: 6 }}>
        <span className={`status-badge ${block.status}`}>
          {STATUS_LABELS[block.status] || block.status}
        </span>
        {block.boundModelId && colors && (
          <span style={{ marginLeft: 4, fontSize: 10, color: colors.accent }}>
            → {block.boundModelId === 'model_mrmodn' ? 'DCPRES' : block.boundModelId}
          </span>
        )}
        {block.boundModelId && !colors && (
          <span style={{ marginLeft: 4, fontSize: 10, color: 'var(--ws-text-muted)' }}>
            → {block.boundModelId === 'model_mrmodn' ? 'DCPRES' : block.boundModelId}
          </span>
        )}
      </div>
    </div>
  );
};

export default SequenceBlockCard;
