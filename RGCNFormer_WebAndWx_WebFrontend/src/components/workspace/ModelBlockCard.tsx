/**
 * ModelBlockCard.tsx - Model Zone 模型卡片 / Model Zone model card
 *
 * Workspace 工作区"Model Zone"中的卡片,代表一个可用的模型(DCPRES 等)。
 * 展示模型标题、描述、状态(available → "✓ Available" / disabled → "Coming Soon")、
 * 版本号,以及已绑定的 sequence id 列表(每条显示为彩色圆点)。
 * disabled 状态下点击不触发选中(disabled 卡 onClick 为 undefined)。
 * Card inside the workspace's "Model Zone" rendering an available model (e.g. DCPRES).
 * Renders title, description, status (available/disabled), version, and the list of
 * bound sequence ids (each as a colored dot). Disabled cards do not respond to clicks.
 *
 * 功能模块 / Modules:
 * - 状态徽章:available(✓)/disabled(Coming Soon)/ Status badge rendering
 * - 绑定序列彩色圆点(BINDING_COLORS 派生)/ Bound-sequence colored dots
 * - 选中态高亮(box-shadow) / Selected highlight
 *
 * 输入 / Inputs:
 * - block: ModelBlock / model block
 * - isSelected: boolean / selection flag
 * - onClick: () => void / click callback (disabled when block.status==='disabled')
 * - boundSequenceIds?: string[] / ids of sequences bound to this model
 *
 * 输出 / Outputs:
 * - JSX.Element 卡片 / Card JSX
 *
 * 数据流 / Data Flow:
 * 1. WorkspacePage 注入 modelBlocks(MODEL_REGISTRY 派生)
 * 2. WorkspaceCanvas 计算每个 model 的 boundSequenceIds(扫 sequenceBlocks)
 * 3. 用户点击 → onSelectBlock(block.id) → PropertiesPanel 展示 ModelProperties
 * 4. 状态从 disabled 切换为 available 时,workspace 顶层 model registry 触发重渲染
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: workspace/types(BINDING_COLORS, getBindingColorIndex)
 * - 被调用 / Called by: components/workspace/WorkspaceCanvas.tsx
 * - 关联 / Related: SequenceBlockCard(共享颜色)、PropertiesPanel(ModelProperties)
 *
 * 使用示例 / Usage Example:
 *   <ModelBlockCard
 *     block={modelBlock}
 *     isSelected={selectedId === modelBlock.id}
 *     onClick={() => onSelectBlock(modelBlock.id)}
 *     boundSequenceIds={[seq1.id, seq2.id]}
 *   />
 */

import React from 'react';
import type { ModelBlock } from './types';
import { BINDING_COLORS, getBindingColorIndex } from './types';

interface ModelBlockCardProps {
  block: ModelBlock;
  isSelected: boolean;
  onClick: () => void;
  boundSequenceIds?: string[];
}

const ModelBlockCard: React.FC<ModelBlockCardProps> = ({
  block,
  isSelected,
  onClick,
  boundSequenceIds = [],
}) => {
  const isDisabled = block.status === 'disabled';

  // Binding color based on own model id (same model = same color across zones)
  const colorIdx = getBindingColorIndex(block.id);
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
      className={`block-card ${isSelected ? 'selected' : ''} ${isDisabled ? 'disabled' : ''}`}
      onClick={isDisabled ? undefined : onClick}
      style={cardStyle}
    >
      <div className="block-card-title">
        🧠 {block.title}
      </div>
      <div className="block-card-subtitle">
        {block.description.slice(0, 60)}...
      </div>
      <div className="model-card-status">
        {isDisabled ? (
          <span className="coming-soon-badge">Coming Soon</span>
        ) : (
          <span className={`status-badge ${block.status}`}>
            {block.status === 'available' ? '✓ Available' : block.status}
          </span>
        )}
      </div>
      <div className="block-card-meta" style={{ marginTop: 4 }}>
        Version: {block.version}
      </div>
      {boundSequenceIds.length > 0 && (
        <div style={{ display: 'flex', gap: 4, marginTop: 6, alignItems: 'center' }}>
          <span style={{ fontSize: 10, color: 'var(--ws-text-muted)' }}>Bound:</span>
          {boundSequenceIds.map((seqId) => (
            <span
              key={seqId}
              style={{
                display: 'inline-block',
                width: 10,
                height: 10,
                borderRadius: '50%',
                background: colors ? colors.accent : '#ccc',
              }}
              title={seqId}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default ModelBlockCard;
