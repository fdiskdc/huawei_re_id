# Draft: 添加 DSCPS 和 ProCSE 模型到 Model Zone

## Requirements (confirmed)
- 在 Model Zone 添加 DSCPS 模型和 ProCSE 模型
- 这两个模型后台都调用已有的 DCPRES 模型

## Technical Decisions
- 需要在 mockData.ts 的 DEFAULT_MODELS 数组中添加两个新模型
- 模型状态设为 'available'
- 前端展示使用相同的 ModelBlockCard 组件

## Research Findings
### 现有项目结构
- **模型定义**: `src/components/workspace/mockData.ts` - DEFAULT_MODELS 数组
- **模型类型**: `src/components/workspace/types.ts` - ModelBlock 接口
- **模型卡片**: `src/components/workspace/ModelBlockCard.tsx` - 展示组件
- **API 调用**: `src/lib/api.ts` - submitTask 函数
- **国际化**: `src/lib/i18n/zh.ts` 和 `en.ts`

### 现有模型
1. DCPRES (id: model_rgcnformer)
2. GCN (id: model_gcn)
3. K-Means (id: model_kmeans)

### ModelBlock 接口
```typescript
interface ModelBlock {
  id: string;
  type: 'model';
  title: string;
  modelName: string;
  status: ModelStatus; // 'available' | 'disabled' | 'loading'
  description: string;
  version: string;
}
```

## Confirmed Requirements
1. **DSCPS 描述**: "Deep Semi-Supervised Clustering based on Pairwise Constraints and Sample Similarity"
2. **ProCSE 描述**: "Progressive Contrastive Structural Entropy for Graph Clustering"
3. **版本号**: v1.0
4. **后台调用方式**: 前端只展示，后端自动路由到 DCPRES
5. **初始状态**: available (立即可用)

## Scope Boundaries
- INCLUDE: 添加模型定义到 mockData.ts
- INCLUDE: 添加国际化翻译
- EXCLUDE: 后端 API 修改
- EXCLUDE: 修改现有 DCPRES 模型逻辑
