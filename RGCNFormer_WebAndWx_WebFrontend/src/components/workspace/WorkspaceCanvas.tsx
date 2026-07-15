/**
 * WorkspaceCanvas.tsx - 工作区中央画布(三区布局)/ Workspace central canvas
 *
 * WorkspacePage 中央区域,纵向分为三个 zone:
 *   1. Input Zone: 渲染 sequenceBlocks(SequenceBlockCard 列表)
 *   2. Model Zone: 渲染 modelBlocks(ModelBlockCard 列表,每个 model 携带 boundSequenceIds)
 *   3. Visualization Zone: 渲染 vizBlocks(VizBlockCard 列表,每个 viz 携带 boundSequenceLabel)
 * 用户点击任意卡片触发 onSelectBlock(id) → PropertiesPanel 展示该 block 的属性。
 * Center area of WorkspacePage, vertically divided into three zones:
 *   1. Input Zone: renders sequenceBlocks via SequenceBlockCard list
 *   2. Model Zone: renders modelBlocks via ModelBlockCard (with boundSequenceIds per model)
 *   3. Visualization Zone: renders vizBlocks via VizBlockCard (with boundSequenceLabel per viz)
 * Clicking any card fires onSelectBlock(id) → PropertiesPanel shows that block's properties.
 *
 * 功能模块 / Modules:
 * - Input Zone 渲染(空态:引导点击 PuzzleLibrary)/ Input Zone render with empty hint
 * - Model Zone 渲染(计算 boundSequenceIds 列表)/ Model Zone render + boundSequenceIds calc
 * - Visualization Zone 渲染(查找 boundSequenceLabel)/ Visualization Zone render + label calc
 * - 选中态高亮(对比 selectedBlockId)/ Selected highlight by id
 *
 * 输入 / Inputs:
 * - sequenceBlocks / modelBlocks / vizBlocks: 三类 block 列表
 * - selectedBlockId: string | null / currently selected block id
 * - onSelectBlock: (id: string) => void / selection callback
 *
 * 输出 / Outputs:
 * - JSX.Element 三区画布 / Three-zone canvas JSX
 *
 * 数据流 / Data Flow:
 * 1. WorkspacePage 维护三类 block state
 * 2. 把列表 + selectedBlockId 传入 WorkspaceCanvas
 * 3. 派生每个 model 的 boundSequenceIds(scan sequenceBlocks)
 * 4. 派生每个 viz 的 boundSequenceLabel(find sequenceBlocks by id)
 * 5. 用户点击 → onSelectBlock(id) → 父组件更新 selectedBlockId
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: SequenceBlockCard, ModelBlockCard, VizBlockCard
 * - 被调用 / Called by: pages/WorkspacePage.tsx
 * - 关联 / Related: PuzzleLibrary(添加 block 入口)、PropertiesPanel(选中后展示属性)
 *
 * 使用示例 / Usage Example:
 *   <WorkspaceCanvas
 *     sequenceBlocks={sequenceBlocks}
 *     modelBlocks={modelBlocks}
 *     vizBlocks={vizBlocks}
 *     selectedBlockId={selectedBlockId}
 *     onSelectBlock={setSelectedBlockId}
 *   />
 */

import React from 'react';
import type { SequenceBlock, ModelBlock, VisualizationBlock } from './types';
import SequenceBlockCard from './SequenceBlockCard';
import ModelBlockCard from './ModelBlockCard';
import VizBlockCard from './VizBlockCard';

interface WorkspaceCanvasProps {
  sequenceBlocks: SequenceBlock[];
  modelBlocks: ModelBlock[];
  vizBlocks: VisualizationBlock[];
  selectedBlockId: string | null;
  onSelectBlock: (id: string) => void;
}

const WorkspaceCanvas: React.FC<WorkspaceCanvasProps> = ({
  sequenceBlocks,
  modelBlocks,
  vizBlocks,
  selectedBlockId,
  onSelectBlock,
}) => {
  return (
    <div className="workspace-canvas">
      {/* Input Zone */}
      <div className="canvas-zone zone-input">
        <div className="canvas-zone-label">
          <span className="zone-dot" />
          Input Zone
        </div>
        <div className="canvas-zone-content">
          {sequenceBlocks.length === 0 ? (
            <div className="zone-empty">
              Click a dataset (Human/Plant/3Gen) in the puzzle library to add a sequence block
            </div>
          ) : (
            sequenceBlocks.map((block) => (
              <SequenceBlockCard
                key={block.id}
                block={block}
                isSelected={selectedBlockId === block.id}
                onClick={() => onSelectBlock(block.id)}
              />
            ))
          )}
        </div>
      </div>

      {/* Model Zone */}
      <div className="canvas-zone zone-model">
        <div className="canvas-zone-label">
          <span className="zone-dot" />
          Model Zone
        </div>
        <div className="canvas-zone-content">
          {modelBlocks.map((block) => {
            const boundSeqIds = sequenceBlocks
              .filter((s) => s.boundModelId === block.id)
              .map((s) => s.id);
            return (
              <ModelBlockCard
                key={block.id}
                block={block}
                isSelected={selectedBlockId === block.id}
                onClick={() => onSelectBlock(block.id)}
                boundSequenceIds={boundSeqIds}
              />
            );
          })}
        </div>
      </div>

      {/* Visualization Zone */}
      <div className="canvas-zone zone-visualization">
        <div className="canvas-zone-label">
          <span className="zone-dot" />
          Visualization Zone
        </div>
        <div className="canvas-zone-content">
          {vizBlocks.length === 0 ? (
            <div className="zone-empty">
              Click visualization blocks in the puzzle library to add them here
            </div>
          ) : (
            vizBlocks.map((block) => {
              const boundSeq = sequenceBlocks.find((s) => s.id === block.boundSequenceId);
              return (
                <VizBlockCard
                  key={block.id}
                  block={block}
                  isSelected={selectedBlockId === block.id}
                  onClick={() => onSelectBlock(block.id)}
                  modelBlocks={modelBlocks}
                  boundSequenceLabel={boundSeq ? `${boundSeq.title} (${boundSeq.dataset})` : undefined}
                />
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};

export default WorkspaceCanvas;