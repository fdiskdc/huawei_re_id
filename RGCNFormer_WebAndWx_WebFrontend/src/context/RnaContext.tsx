/**
 * RnaContext.tsx - RNA 全局状态 Context(序列、服务器、数据集)/ RNA global state context
 *
 * React Context,集中管理跨页面共享的 RNA 相关状态:
 *   - rnaSequence / setRnaSequence: 用户输入的 RNA 序列
 *   - server / setServer: 后端服务器选择(主备切换)
 *   - dataset / setDataset / datasetIndex: 数据集类型(Human/Plant/3Gen 等)与下拉索引
 *   - jobId / setJobId: 后端返回的当前任务 ID
 *   - resultData / setResultData: 后端推理结果(分类/注意力/IG/UMAP 等)
 *   - 各种 setter 透传给 useRna() hook
 * React Context that centralizes RNA-related state shared across pages:
 *   - rnaSequence / setRnaSequence: user-input RNA sequence
 *   - server / setServer: backend server selection (primary/backup failover)
 *   - dataset / setDataset / datasetIndex: dataset type & dropdown index
 *   - jobId / setJobId: current job id returned by the backend
 *   - resultData / setResultData: backend inference results (classification/attention/IG/UMAP)
 *   - additional setters exposed via the useRna() hook
 *
 * 功能模块 / Modules:
 * - ClassificationResult: 分类结果类型(单项 name/value)/ Classification result type
 * - RnaContextType: 状态+setter 集合的 TypeScript 接口 / RnaContextType interface
 * - RnaProvider: Context.Provider,持有 useState 并通过 value 透出 / Context provider
 * - useRna(): 自定义 hook,封装 useContext,未在 Provider 内时抛错 / useRna() custom hook
 *
 * 输入 / Inputs:
 * - children: ReactNode(由 main.tsx 中 <RnaProvider>{children}</RnaProvider> 包裹)
 *   / React children wrapped by main.tsx
 *
 * 输出 / Outputs:
 * - RnaContext.Provider 暴露的 value(state + setters)/ Exposed value (state + setters)
 *
 * 数据流 / Data Flow:
 * 1. main.tsx 装配 <RnaProvider> → 包裹整个路由树
 * 2. 任意 Page 通过 const { rnaSequence, setRnaSequence } = useRna() 获取/更新状态
 * 3. 用户在 LocalizationViz 输入序列 → setRnaSequence
 * 4. 用户点击"提交" → 调 lib/api.ts → 后端返回 jobId → setJobId
 * 5. 轮询结果 → setResultData → 其它 Page 通过 useRna() 读取
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: lib/api.ts(DatasetType 类型)、lib/i18n(LanguageContext 独立,无依赖)
 * - 被调用 / Called by: main.tsx(<RnaProvider> 装配)、多个 Page(consume)
 * - 关联 / Related: lib/api.ts(后端调用结果回填 setResultData)
 *
 * 使用示例 / Usage Example:
 *   // main.tsx
 *   <RnaProvider><App /></RnaProvider>
 *   // LocalizationViz.tsx
 *   const { rnaSequence, setRnaSequence, submit } = useRna();
 *   setRnaSequence('ACGU...');
 */
/*
 * @Author: Chao Deng && chaodeng987@outlook.com
 * @Date: 2026-01-20 12:06:34
 * @LastEditors: Chao Deng && chaodeng987@outlook.com
 * @LastEditTime: 2026-01-20 16:45:57
 * @FilePath: /mrmodn_mobile_web/frontend/src/context/RnaContext.tsx
 * @Description: 
 * 那只是一场游戏一场梦
 *  
 * https://orcid.org/0009-0009-8520-1656
 * DOI: 10.3390/app15158626
 * DOI: 10.3390/rs17142354
 * Copyright (c) 2026 by ${Chao Deng}, All Rights Reserved. 
 */
import React, { createContext, useState, useContext, type ReactNode } from 'react';
import type { DatasetType } from '../lib/api';

export interface ClassificationResult {
    name: string;
    value: number;
}

interface RnaContextType {
    rnaSequence: string;
    setRnaSequence: (sequence: string) => void;
    server: string;
    setServer: (server: string) => void;
    dataset: DatasetType;
    setDataset: (dataset: DatasetType) => void;
    datasetIndex: number;
    setDatasetIndex: (index: number) => void;
    classificationResults: ClassificationResult[];
    setClassificationResults: (results: ClassificationResult[]) => void;
}

const RnaContext = createContext<RnaContextType | undefined>(undefined);

export const RnaProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [rnaSequence, setRnaSequence] = useState('');
    const [server, setServer] = useState('');
    const [dataset, setDataset] = useState<DatasetType>('Human');
    const [datasetIndex, setDatasetIndex] = useState(0);
    const [classificationResults, setClassificationResults] = useState<ClassificationResult[]>([]);

    return (
        <RnaContext.Provider value={{ rnaSequence, setRnaSequence, server, setServer, dataset, setDataset, datasetIndex, setDatasetIndex, classificationResults, setClassificationResults }}>
            {children}
        </RnaContext.Provider>
    );
};

export const useRna = () => {
    const context = useContext(RnaContext);
    if (!context) {
        throw new Error('useRna must be used within a RnaProvider');
    }
    return context;
};
