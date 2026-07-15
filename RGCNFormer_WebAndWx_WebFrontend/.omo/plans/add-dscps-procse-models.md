# 添加 DSCPS 和 ProCSE 模型到 Model Zone

## TL;DR

> **Quick Summary**: 在前端 Model Zone 添加 DSCPS 和 ProCSE 两个模型卡片，后台自动路由到 DCPRES 处理
>
> **Deliverables**:
> - mockData.ts 中新增两个模型定义
> - i18n 文件中添加相关翻译（如需要）
>
> **Estimated Effort**: Quick
> **Parallel Execution**: NO - 单文件修改
> **Critical Path**: Task 1 → Done

---

## Context

### Original Request
用户希望在 RGCNFormer_WebAndWx_WebFrontend 的 Model Zone 添加 DSCPS 和 ProCSE 两个模型，后台调用已有的 DCPRES 模型。

### Interview Summary
**Key Discussions**:
- DSCPS 描述: "Deep Semi-Supervised Clustering based on Pairwise Constraints and Sample Similarity"
- ProCSE 描述: "Progressive Contrastive Structural Entropy for Graph Clustering"
- 版本号: v1.0
- 前端只负责展示，后端自动路由到 DCPRES
- 状态: available (立即可用)

---

## Work Objectives

### Core Objective
在 Model Zone 的模型列表中添加 DSCPS 和 ProCSE 两个模型选项

### Concrete Deliverables
- `src/components/workspace/mockData.ts` - 添加两个模型定义

### Definition of Done
- [ ] Model Zone 显示 5 个模型：DCPRES、GCN、K-Means、DSCPS、ProCSE
- [ ] 新模型可点击选择
- [ ] 新模型显示正确的描述和版本

### Must Have
- 模型 ID 命名规范：`model_dscps`、`model_procse`
- 模型名称：`DSCPS`、`ProCSE`
- 状态：`available`
- 版本：`v1.0`

### Must NOT Have (Guardrails)
- 不修改现有 DCPRES、GCN、K-Means 模型
- 不修改后端 API 逻辑
- 不添加不必要的复杂性

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: None
- **Agent-Executed QA**: YES - 视觉验证

---

## TODOs

- [ ] 1. 在 mockData.ts 中添加 DSCPS 和 ProCSE 模型定义

  **What to do**:
  - 打开 `src/components/workspace/mockData.ts`
  - 在 `DEFAULT_MODELS` 数组中添加两个新模型对象
  - DSCPS 模型:
    ```typescript
    {
      id: 'model_dscps',
      type: 'model',
      title: 'DSCPS',
      modelName: 'DSCPS',
      status: 'available',
      description: 'Deep Semi-Supervised Clustering based on Pairwise Constraints and Sample Similarity',
      version: 'v1.0',
    }
    ```
  - ProCSE 模型:
    ```typescript
    {
      id: 'model_procse',
      type: 'model',
      title: 'ProCSE',
      modelName: 'ProCSE',
      status: 'available',
      description: 'Progressive Contrastive Structural Entropy for Graph Clustering',
      version: 'v1.0',
    }
    ```

  **Must NOT do**:
  - 不修改现有模型定义
  - 不改变数组顺序（新模型追加到末尾）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单文件简单修改，无需复杂逻辑
  - **Skills**: []
    - 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: None (可立即开始)

  **References**:

  **Pattern References**:
  - `src/components/workspace/mockData.ts:16-47` - 现有 DEFAULT_MODELS 数组结构，照此格式添加

  **API/Type References**:
  - `src/components/workspace/types.ts:47-55` - ModelBlock 接口定义

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 验证新模型已添加
    Tool: Bash (grep)
    Preconditions: 无
    Steps:
      1. 运行 grep -n "model_dscps\|model_procse" src/components/workspace/mockData.ts
      2. 确认输出包含两个模型的 ID
    Expected Result: 找到 model_dscps 和 model_procse 两个 ID
    Evidence: .omo/evidence/task-1-model-added.txt

  Scenario: 验证模型数量
    Tool: Bash (grep)
    Preconditions: 无
    Steps:
      1. 运行 grep -c "type: 'model'" src/components/workspace/mockData.ts
      2. 确认数量为 5
    Expected Result: 输出 5
    Evidence: .omo/evidence/task-1-model-count.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add DSCPS and ProCSE models to Model Zone`
  - Files: `src/components/workspace/mockData.ts`

---

## Success Criteria

### Verification Commands
```bash
grep -n "model_dscps\|model_procse" src/components/workspace/mockData.ts
# Expected: 两个模型 ID 出现在输出中

grep -c "type: 'model'" src/components/workspace/mockData.ts
# Expected: 5
```

### Final Checklist
- [ ] DSCPS 模型已添加，ID 为 model_dscps
- [ ] ProCSE 模型已添加，ID 为 model_procse
- [ ] 两个模型状态均为 available
- [ ] 两个模型版本均为 v1.0
- [ ] 描述文本正确
- [ ] 现有模型未被修改
