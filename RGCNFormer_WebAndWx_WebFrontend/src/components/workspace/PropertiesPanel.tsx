/**
 * PropertiesPanel.tsx - 右栏:选中 block 的可编辑属性 / Right sidebar block properties
 *
 * WorkspacePage 右侧栏,根据当前选中 block 的 type 渲染不同的属性表单:
 *   - SequenceProperties:标题、数据集、序列输入(过滤 ACGU,最长 1001)、绑定模型、状态、Job ID
 *   - ModelProperties:模型名、版本、状态、描述(无编辑)
 *   - VisualizationProperties:viz 类型、绑定数据源(sequence block)、绑定模型、参数
 *     (Top X / Target Class ID / Top N / Target Node Index,视 viz 类型而异)
 * SequenceProperties 在首次挂载时若序列为空,自动 fetchSampleSequence 拉取示例。
 * Right sidebar of WorkspacePage. Renders one of three property forms based on the selected
 * block's type:
 *   - SequenceProperties: title, dataset, sequence input (ACGU filter, max 1001),
 *     bound model, status, job id
 *   - ModelProperties: model name, version, status, description (read-only)
 *   - VisualizationProperties: viz type, bound data source (sequence block), bound model,
 *     and per-type parameters (Top X / Target Class ID / Top N / Target Node Index)
 * SequenceProperties auto-fetches a sample sequence on first mount if the sequence is empty.
 *
 * 功能模块 / Modules:
 * - 三种 block 类型的子属性组件 / Three per-type sub-components
 * - 关闭按钮 (右上角 ✕)/ Close button (✕)
 * - Remove / Apply 动作按钮 / Remove / Apply action buttons
 * - 自动拉取示例序列(SequenceProperties)/ Auto-fetch sample sequence
 *
 * 输入 / Inputs:
 * - block: WorkspaceBlock | null / currently selected block (null = empty state)
 * - sequenceBlocks / modelBlocks / vizBlocks 列表 / workspace state
 * - onUpdateBlock / onRemoveBlock / onRunViz 回调 / parent callbacks
 *
 * 输出 / Outputs:
 * - JSX.Element 属性面板 / Properties panel JSX
 *
 * 数据流 / Data Flow:
 * 1. 用户在 WorkspaceCanvas 点击 block → onSelectBlock(id)
 * 2. WorkspacePage 把该 block 传入 <PropertiesPanel block=...>
 * 3. 渲染对应子组件(SequenceProperties / ModelProperties / VisualizationProperties)
 * 4. 用户修改 → onUpdateBlock → WorkspacePage 更新 state → 卡片重渲染
 * 5. 用户点击 Apply → onRunViz(blockId) → 调用后端 API
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: lib/api.ts(fetchSampleSequence)、workspace/types(VIZ_TYPE_REGISTRY 等)
 * - 被调用 / Called by: pages/WorkspacePage.tsx
 * - 关联 / Related: WorkspaceCanvas(选中交互)、VizBlockCard/SequenceBlockCard/ModelBlockCard
 *
 * 使用示例 / Usage Example:
 *   <PropertiesPanel
 *     block={selectedBlock}
 *     sequenceBlocks={sequenceBlocks}
 *     modelBlocks={modelBlocks}
 *     onUpdateBlock={(id, upd) => updateBlock(id, upd)}
 *     onRemoveBlock={(id) => removeBlock(id)}
 *     onRunViz={(id) => runVisualization(id)}
 *   />
 */

import React from 'react';
import type { WorkspaceBlock, SequenceBlock, ModelBlock, VisualizationBlock, DatasetType } from './types';
import { VIZ_TYPE_REGISTRY, DATASET_OPTIONS, DATASET_LABELS } from './types';
import { fetchSampleSequence } from '../../lib/api';

interface PropertiesPanelProps {
  block: WorkspaceBlock | null;
  sequenceBlocks: SequenceBlock[];
  modelBlocks: ModelBlock[];
  onUpdateBlock: (id: string, updates: Partial<WorkspaceBlock>) => void;
  onRemoveBlock: (id: string) => void;
  onRunViz: (blockId: string) => void;
}

const PropertiesPanel: React.FC<PropertiesPanelProps> = ({
  block,
  sequenceBlocks,
  modelBlocks,
  onUpdateBlock,
  onRemoveBlock,
  onRunViz,
}) => {
  if (!block) {
    return (
      <div className="properties-panel">
        <div className="properties-empty">
          <div className="properties-empty-icon">📋</div>
          <div>Select a block on the canvas<br />to view its properties</div>
        </div>
      </div>
    );
  }

  return (
    <div className="properties-panel">
      <div className="properties-panel-header">
        <h3>Properties</h3>
        <button
          className="properties-panel-close"
          onClick={() => onUpdateBlock(block.id, {})}
          title="Close panel"
        >
          ✕
        </button>
      </div>
      <div className="properties-panel-body">
        {block.type === 'sequence' && (
          <SequenceProperties
            block={block}
            modelBlocks={modelBlocks}
            onUpdate={(updates) => onUpdateBlock(block.id, updates)}
            onRemove={() => onRemoveBlock(block.id)}
          />
        )}
        {block.type === 'model' && (
          <ModelProperties
            block={block}
            onRemove={() => onRemoveBlock(block.id)}
          />
        )}
        {block.type === 'visualization' && (
          <VisualizationProperties
            block={block}
            sequenceBlocks={sequenceBlocks}
            modelBlocks={modelBlocks}
            onUpdate={(updates) => onUpdateBlock(block.id, updates)}
            onRemove={() => onRemoveBlock(block.id)}
            onRun={() => onRunViz(block.id)}
          />
        )}
      </div>
    </div>
  );
};

// ==================== Sequence Properties ====================

interface SequencePropertiesProps {
  block: SequenceBlock;
  modelBlocks: ModelBlock[];
  onUpdate: (updates: Partial<SequenceBlock>) => void;
  onRemove: () => void;
}

const SequenceProperties: React.FC<SequencePropertiesProps> = ({
  block,
  modelBlocks,
  onUpdate,
  onRemove,
}) => {

  React.useEffect(() => {
    if (!block.sequence) {
      fetchSampleSequence()
        .then((data) => {
          if (data.sequence) {
            onUpdate({ sequence: data.sequence.toUpperCase() });
          }
        })
        .catch(() => {});
    }
  }, []);

  return (
    <>
      {/* Title */}
      <div className="prop-group">
        <label className="prop-label">Title</label>
        <input
          className="prop-input"
          value={block.title}
          onChange={(e) => onUpdate({ title: e.target.value })}
        />
      </div>

      {/* Dataset */}
      <div className="prop-group">
        <label className="prop-label">Dataset</label>
        <select
          className="prop-select"
          value={block.dataset}
          onChange={(e) => onUpdate({ dataset: e.target.value as DatasetType })}
        >
          {DATASET_OPTIONS.map((d) => (
            <option key={d} value={d}>
              {DATASET_LABELS[d]}
            </option>
          ))}
        </select>
      </div>

      {/* Dataset Size */}
      <div className="prop-group">
        <label className="prop-label">Sequences in Dataset</label>
        <div className="prop-value">{block.sequenceCount.toLocaleString()}</div>
      </div>

      <div className="prop-divider" />

      {/* Sequence Input */}
      <div className="prop-group">
        <label className="prop-label">RNA 序列 (ACGU)</label>
        <textarea
          className="prop-textarea"
          value={block.sequence}
          placeholder="输入 RNA 序列（仅允许 A/C/G/U，最长 1001）..."
          maxLength={1001}
          rows={4}
          style={{
            fontFamily: "'Courier New', Courier, monospace",
            fontSize: 12,
            lineHeight: 1.4,
          }}
          onChange={(e) => {
            const filtered = e.target.value.toUpperCase().replace(/[^ACGU]/g, '');
            onUpdate({ sequence: filtered });
          }}
        />
        <div className="prop-char-counts">
          {(['A', 'C', 'G', 'U'] as const).map((ch) => (
            <span key={ch} className={`prop-char-tag ${ch}`}>
              {ch}: {block.sequence.split('').filter((c) => c === ch).length}
            </span>
          ))}
          <span className="prop-char-tag" style={{ background: '#ede9e5', color: '#7d7872' }}>
            长度: {block.sequence.length}/1001
          </span>
        </div>
      </div>

      <div className="prop-divider" />

      {/* Bind Model */}
      <div className="prop-group">
        <label className="prop-label">Bind Model</label>
        <select
          className="prop-select"
          value={block.boundModelId || ''}
          onChange={(e) => onUpdate({ boundModelId: e.target.value || null })}
        >
          <option value="">-- Select Model --</option>
          {modelBlocks.map((m) => (
            <option key={m.id} value={m.id} disabled={m.status === 'disabled'}>
              {m.title} {m.status === 'disabled' ? '(Coming Soon)' : ''}
            </option>
          ))}
        </select>
      </div>

      {/* Status */}
      <div className="prop-group">
        <label className="prop-label">Status</label>
        <div className="prop-value">
          <span className={`status-badge ${block.status}`}>
            {block.status}
          </span>
        </div>
      </div>

      {/* Job ID */}
      {block.jobId && (
        <div className="prop-group">
          <label className="prop-label">Job ID</label>
          <div className="prop-value" style={{ fontSize: 11, wordBreak: 'break-all' }}>
            {block.jobId}
          </div>
        </div>
      )}

      <div className="prop-divider" />

      {/* Actions */}
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          className="prop-btn"
          style={{ background: '#f0dede', color: 'var(--ws-status-error)' }}
          onClick={onRemove}
        >
          Remove
        </button>
      </div>

      {/* Result Summary */}
      {block.resultSummary && (
        <>
          <div className="prop-divider" />
          <div className="prop-group">
            <label className="prop-label">Result Summary</label>
            <div className="prop-value" style={{ fontSize: 12 }}>
              ✓ Classification data available
              <br />
              ✓ Attention data available
              <br />
              ✓ GCN data available
            </div>
          </div>
        </>
      )}
    </>
  );
};

// ==================== Model Properties ====================

interface ModelPropertiesProps {
  block: ModelBlock;
  onRemove?: () => void;
}

const ModelProperties: React.FC<ModelPropertiesProps> = ({ block }) => {
  return (
    <>
      <div className="prop-group">
        <label className="prop-label">Model Name</label>
        <div className="prop-value">{block.modelName}</div>
      </div>

      <div className="prop-group">
        <label className="prop-label">Version</label>
        <div className="prop-value">{block.version}</div>
      </div>

      <div className="prop-group">
        <label className="prop-label">Status</label>
        <div className="prop-value">
          {block.status === 'available' ? (
            <span className="status-badge available">✓ Available</span>
          ) : (
            <span className="coming-soon-badge">Coming Soon</span>
          )}
        </div>
      </div>

      <div className="prop-group">
        <label className="prop-label">Description</label>
        <div className="prop-value" style={{ fontSize: 12, lineHeight: 1.5 }}>
          {block.description}
        </div>
      </div>

      {block.status === 'disabled' && (
        <div
          style={{
            padding: '10px 12px',
            background: '#ede9e5',
            borderRadius: 'var(--ws-radius-sm)',
            fontSize: 12,
            color: 'var(--ws-text-muted)',
            fontStyle: 'italic',
          }}
        >
          This model is not yet available. It will be enabled in a future release.
        </div>
      )}
    </>
  );
};

// ==================== Visualization Properties ====================

interface VisualizationPropertiesProps {
  block: VisualizationBlock;
  sequenceBlocks: SequenceBlock[];
  modelBlocks: ModelBlock[];
  onUpdate: (updates: Partial<VisualizationBlock>) => void;
  onRemove: () => void;
  onRun: () => void;
}

const VisualizationProperties: React.FC<VisualizationPropertiesProps> = ({
  block,
  sequenceBlocks,
  modelBlocks,
  onUpdate,
  onRemove,
  onRun,
}) => {
  const meta = VIZ_TYPE_REGISTRY.find((v) => v.key === block.vizType);

  return (
    <>
      {/* Viz Type */}
      <div className="prop-group">
        <label className="prop-label">Visualization Type</label>
        <div className="prop-value">
          {meta?.icon} {meta?.label || block.vizType}
        </div>
      </div>

      {/* Description */}
      <div className="prop-group">
        <label className="prop-label">Description</label>
        <div className="prop-value" style={{ fontSize: 12, lineHeight: 1.5 }}>
          {meta?.description}
        </div>
      </div>

      <div className="prop-divider" />

      {/* Bind Sequence (Data Source) */}
      <div className="prop-group">
        <label className="prop-label">Data Source (Sequence)</label>
        <select
          className="prop-select"
          value={block.boundSequenceId || ''}
          onChange={(e) => onUpdate({ boundSequenceId: e.target.value || null })}
        >
          <option value="">-- Select Sequence Block --</option>
          {sequenceBlocks.map((s) => (
            <option key={s.id} value={s.id}>
              {s.title} ({s.dataset})
            </option>
          ))}
        </select>
        {sequenceBlocks.length === 0 && (
          <div style={{ fontSize: 11, color: 'var(--ws-text-muted)', marginTop: 4 }}>
            No sequence blocks available. Add one from the puzzle library first.
          </div>
        )}
      </div>

      {/* Bind Model */}
      <div className="prop-group">
        <label className="prop-label">Model</label>
        <select
          className="prop-select"
          value={block.boundModelId || ''}
          onChange={(e) => onUpdate({ boundModelId: e.target.value || null })}
        >
          <option value="">-- Select Model --</option>
          {modelBlocks.map((m) => (
            <option key={m.id} value={m.id} disabled={m.status === 'disabled'}>
              {m.title} {m.status === 'disabled' ? '(Coming Soon)' : ''}
            </option>
          ))}
        </select>
      </div>

      <div className="prop-divider" />

      {/* Parameters */}
      {block.vizType === 'attention' && (
        <div className="prop-group">
          <label className="prop-label">Top X</label>
          <input
            className="prop-input"
            type="number"
            min={1}
            max={100}
            value={block.params.topX || 3}
            onChange={(e) =>
              onUpdate({ params: { ...block.params, topX: parseInt(e.target.value) || 3 } })
            }
          />
        </div>
      )}

      {block.vizType === 'integrated-gradients' && (
        <>
          <div className="prop-group">
            <label className="prop-label">Target Class ID</label>
            <select
              className="prop-select"
              value={block.params.targetClassId ?? 0}
              onChange={(e) =>
                onUpdate({
                  params: { ...block.params, targetClassId: parseInt(e.target.value) },
                })
              }
            >
              {[
                'Am (0)', 'Atol (1)', 'Cm (2)', 'Gm (3)', 'Tm (4)', 'Y (5)',
                'ac4C (6)', 'm1A (7)', 'm5C (8)', 'm6A (9)', 'm6Am (10)', 'm7G (11)',
              ].map((name, idx) => (
                <option key={idx} value={idx}>
                  {name}
                </option>
              ))}
            </select>
          </div>
          <div className="prop-group">
            <label className="prop-label">Top N Nodes</label>
            <input
              className="prop-input"
              type="number"
              min={1}
              max={100}
              value={block.params.topN || 10}
              onChange={(e) =>
                onUpdate({ params: { ...block.params, topN: parseInt(e.target.value) || 10 } })
              }
            />
          </div>
        </>
      )}

      {block.vizType === 'gcn-message-passing' && (
        <div className="prop-group">
          <label className="prop-label">Target Node Index</label>
          <input
            className="prop-input"
            type="number"
            min={0}
            value={block.params.targetNodeIdx ?? 0}
            onChange={(e) =>
              onUpdate({
                params: { ...block.params, targetNodeIdx: parseInt(e.target.value) || 0 },
              })
            }
          />
        </div>
      )}

      {/* Status */}
      <div className="prop-group">
        <label className="prop-label">Status</label>
        <div className="prop-value">
          <span className={`status-badge ${block.status}`}>
            {block.status}
          </span>
        </div>
      </div>

      {/* Result */}
      {block.result && (
        <div className="prop-group">
          <label className="prop-label">Result</label>
          <div className="prop-value" style={{ fontSize: 12 }}>
            ✓ Data available ({JSON.stringify(block.result).length} bytes)
          </div>
        </div>
      )}


      <div className="prop-divider" />

      {/* Actions */}
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          className="prop-btn prop-btn-primary"
          onClick={onRun}
          disabled={!block.boundSequenceId || block.status === 'running'}
        >
          {block.status === 'running' ? 'Running...' : 'Apply'}
        </button>
        <button
          className="prop-btn"
          style={{ background: '#f0dede', color: 'var(--ws-status-error)' }}
          onClick={onRemove}
        >
          Remove
        </button>
      </div>
    </>
  );
};

export default PropertiesPanel;