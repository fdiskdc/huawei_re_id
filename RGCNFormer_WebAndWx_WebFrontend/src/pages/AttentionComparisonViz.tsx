/**
 * AttentionComparisonViz.tsx - 多模型注意力对比 / Multi-model attention comparison
 *
 * /viz-display 之一(也可独立访问)。使用 ECharts 并排展示 mRModN / MultiRM /
 * modX / EvoRMD 等多个模型在同一样本上的注意力分布(柱状图或热力图)。
 * 数据由 fetchAttentionComparison 拉取,经 useQuery 缓存。
 * One of the /viz-display pages. Uses ECharts to side-by-side display attention
 * distributions of multiple models (mRModN / MultiRM / modX / EvoRMD) on the
 * same sample. Data via fetchAttentionComparison + useQuery.
 *
 * 功能模块 / Modules:
 * - 多 ECharts 实例(每个模型一个)/ Multiple ECharts instances
 * - 莫兰迪色卡(MORANDI 派生自样本名)/ Morandi palette (per-sample)
 * - 样本选择(useState)/ Sample selection
 * - 加载/错误态 / Loading/error states
 *
 * 输入 / Inputs:
 * - 后端 /api/v1/attention-comparison 返回 AttentionComparisonSample[]
 *
 * 输出 / Outputs:
 * - JSX.Element 多模型对比图 / Comparison view JSX
 *
 * 数据流 / Data Flow:
 * 1. useQuery → fetchAttentionComparison()
 * 2. 默认选第一个 sample,渲染多个 ECharts
 * 3. 用户切 sample → 重新渲染
 * 4. hover → tooltip
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: lib/api.ts(fetchAttentionComparison, AttentionComparisonSample)
 * - 被调用 / Called by: App.tsx(<Route path="/viz-display"> 内部)
 * - 关联 / Related: AttentionViz.tsx(单模型)、AttentionDistributionViz.tsx
 *
 * 使用示例 / Usage Example:
 *   const { data } = useQuery({
 *     queryKey: ['attention-comparison'],
 *     queryFn: fetchAttentionComparison,
 *   });
 */
import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import * as echarts from 'echarts';
import { fetchAttentionComparison } from '../lib/api';

const MORANDI = {
  mRModN: '#8DA9C4',
  MultiRM: '#B5838D',
  modX: '#A3B18A',
  EvoRMD: '#DDB892',
  true_site: '#6B705C',
  bg: '#FAFAF8',
  textDark: '#4A4A4A',
};

const MODEL_COLORS: Record<string, string> = {
  mRModN: MORANDI.mRModN,
  MultiRM: MORANDI.MultiRM,
  modX: MORANDI.modX,
  EvoRMD: MORANDI.EvoRMD,
};

const MODEL_DISPLAY_NAMES: Record<string, string> = {
  mRModN: 'DCPRES',
  MultiRM: 'ProCSE',
  modX: 'GCN',
  EvoRMD: 'DSCPS',
};

const getDisplayName = (modelName: string): string => MODEL_DISPLAY_NAMES[modelName] || modelName;

interface AttentionChartProps {
  attention: number[];
  trueSites: number[];
  modelName: string;
  seqLength: number;
}

const AttentionChart: React.FC<AttentionChartProps> = ({
  attention,
  trueSites,
  modelName,
  seqLength,
}) => {
  const chartRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!chartRef.current) return;

    const chart = echarts.init(chartRef.current);
    const x = Array.from({ length: seqLength }, (_, i) => i);
    const color = MODEL_COLORS[modelName] || '#888';

    const option: echarts.EChartsOption = {
      backgroundColor: MORANDI.bg,
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const pos = params[0].dataIndex;
          const val = params[0].value;
          const isSite = trueSites.includes(pos);
          return `Position: ${pos}<br/>Attention: ${val.toFixed(6)}${isSite ? '<br/><b>True Site</b>' : ''}`;
        },
      },
      grid: {
        left: 60,
        right: 20,
        top: 30,
        bottom: 30,
      },
      xAxis: {
        type: 'category',
        data: x,
        axisLabel: {
          color: MORANDI.textDark,
          fontSize: 10,
          interval: 199,
        },
        axisLine: { lineStyle: { color: '#ccc' } },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          color: MORANDI.textDark,
          fontSize: 10,
          formatter: (v: number) => v.toFixed(4),
        },
        splitLine: { lineStyle: { color: '#f0f0f0' } },
      },
      series: [
        {
          type: 'line',
          data: attention,
          smooth: true,
          symbol: 'none',
          lineStyle: { color, width: 1.5 },
          areaStyle: { color, opacity: 0.15 },
        },
      ],
      graphic: trueSites.length > 0 ? [
        {
          type: 'group',
          children: trueSites.map((pos) => ({
            type: 'line',
            shape: {
              x1: 0,
              y1: 0,
              x2: 0,
              y2: -20,
            },
            position: [pos * (chartRef.current!.clientWidth - 80) / (seqLength - 1) + 60, 30],
            style: {
              stroke: MORANDI.true_site,
              lineWidth: 1.5,
              lineDash: [4, 4],
            },
          })),
        },
      ] : undefined,
    };

    chart.setOption(option);

    const handleResize = () => chart.resize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.dispose();
    };
  }, [attention, trueSites, modelName, seqLength]);

  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 4,
        padding: '0 8px',
      }}>
        <span style={{
          fontWeight: 'bold',
          color: MODEL_COLORS[modelName],
          fontSize: 13,
        }}>
          {getDisplayName(modelName)}
        </span>
        <span style={{ fontSize: 11, color: '#888' }}>
          True sites: {trueSites.length}
        </span>
      </div>
      <div ref={chartRef} style={{ width: '100%', height: 150 }} />
    </div>
  );
};

const AttentionComparisonViz: React.FC = () => {
  const [selectedSample, setSelectedSample] = useState(0);

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['attentionComparison'],
    queryFn: fetchAttentionComparison,
  });

  if (isLoading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400, color: '#666' }}>
        Loading attention comparison data...
      </div>
    );
  }

  if (isError) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400, color: '#d32f2f' }}>
        Failed to load attention data: {error?.message}
      </div>
    );
  }

  if (!data || data.samples.length === 0) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400, color: '#666' }}>
        No attention data available
      </div>
    );
  }

  const sample = data.samples[selectedSample];
  const modelNames = data.model_names;

  // Get all active classes across all models for this sample
  const allActiveClasses = new Set<number>();
  Object.values(sample.models).forEach((m) => {
    m.class_indices.forEach((ci) => allActiveClasses.add(ci));
  });
  const activeClassIndices = Array.from(allActiveClasses).sort((a, b) => a - b);

  return (
    <div>
      {/* Sample selector */}
      <div style={{
        display: 'flex',
        gap: 8,
        marginBottom: 16,
        flexWrap: 'wrap',
      }}>
        {data.samples.map((s, i) => (
          <button
            key={i}
            onClick={() => setSelectedSample(i)}
            style={{
              padding: '8px 16px',
              borderRadius: 8,
              border: selectedSample === i ? '2px solid #B8A9C9' : '1px solid #ddd',
              backgroundColor: selectedSample === i ? '#F0EBE6' : '#fff',
              cursor: 'pointer',
              fontWeight: selectedSample === i ? 'bold' : 'normal',
              color: MORANDI.textDark,
            }}
          >
            Sample #{i + 1} (idx: {s.index})
          </button>
        ))}
      </div>

      {/* Sample info */}
      <div style={{
        marginBottom: 16,
        padding: '12px 16px',
        backgroundColor: '#F5F0EB',
        borderRadius: 8,
        fontSize: 13,
        color: MORANDI.textDark,
      }}>
        <b>Sample #{selectedSample + 1}</b> — Active modifications:{' '}
        {activeClassIndices.map((ci) => data.class_names[ci]).join(', ')}
      </div>

      {/* Attention charts per class */}
      {activeClassIndices.map((classIdx) => {
        const className = data.class_names[classIdx];
        return (
          <div key={classIdx} style={{
            marginBottom: 24,
            padding: 16,
            backgroundColor: '#fff',
            borderRadius: 12,
            boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
          }}>
            <h3 style={{
              margin: '0 0 12px 0',
              fontSize: 16,
              fontWeight: 'bold',
              color: MORANDI.textDark,
              borderBottom: '2px solid #B8A9C9',
              paddingBottom: 8,
            }}>
              {className}
            </h3>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: 12,
            }}>
              {modelNames.map((modelName) => {
                const modelData = sample.models[modelName];
                const classIdxInModel = modelData.class_indices.indexOf(classIdx);
                const hasClass = classIdxInModel !== -1;

                return (
                  <AttentionChart
                    key={modelName}
                    attention={hasClass ? modelData.attention[classIdxInModel] : new Array(1001).fill(0)}
                    trueSites={modelData.true_sites}
                    modelName={modelName}
                    seqLength={1001}
                  />
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default AttentionComparisonViz;
