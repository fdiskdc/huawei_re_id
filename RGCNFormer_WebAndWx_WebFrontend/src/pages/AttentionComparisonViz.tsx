/**
 * AttentionComparisonViz.tsx - 多模型注意力对比 / Multi-model attention comparison
 *
 * /viz-display 之一(也可独立访问)。使用 ECharts 并排展示 DCPRES / SCDGC /
 * DSCPS / GCN 在同一样本上的注意力概率密度。
 * 数据由 fetchAttentionComparison 拉取,经 useQuery 缓存。
 * One of the /viz-display pages. Uses ECharts to side-by-side display attention
 * distributions of multiple models (DCPRES / SCDGC / DSCPS / GCN) on the
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
  DCPRES: '#8DA9C4',
  SCDGC: '#B5838D',
  DSCPS: '#A3B18A',
  GCN: '#DDB892',
  true_site: '#6B705C',
  bg: '#FAFAF8',
  textDark: '#4A4A4A',
};

const MODEL_COLORS: Record<string, string> = {
  DCPRES: MORANDI.DCPRES,
  SCDGC: MORANDI.SCDGC,
  DSCPS: MORANDI.DSCPS,
  GCN: MORANDI.GCN,
};

const GAUSSIAN_BANDWIDTH = 12;

/**
 * Apply a Gaussian kernel to discrete attention weights and normalize the
 * result so its integral (with a unit position interval) is one.
 */
const gaussianProbabilityDensity = (values: number[], bandwidth: number): number[] => {
  if (values.length === 0) return [];

  const radius = Math.ceil(bandwidth * 4);
  const kernel = Array.from({ length: radius * 2 + 1 }, (_, index) => {
    const distance = index - radius;
    return Math.exp(-(distance * distance) / (2 * bandwidth * bandwidth));
  });

  const density = values.map((_, position) => {
    let weightedSum = 0;
    const start = Math.max(0, position - radius);
    const end = Math.min(values.length - 1, position + radius);

    for (let sourcePosition = start; sourcePosition <= end; sourcePosition += 1) {
      const kernelWeight = kernel[sourcePosition - position + radius];
      weightedSum += Math.max(0, values[sourcePosition] || 0) * kernelWeight;
    }

    return weightedSum;
  });

  const total = density.reduce((sum, value) => sum + value, 0);
  return total > 0 ? density.map((value) => value / total) : density;
};

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
    const density = gaussianProbabilityDensity(attention, GAUSSIAN_BANDWIDTH);

    const option: echarts.EChartsOption = {
      backgroundColor: MORANDI.bg,
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const pos = params[0].value[0];
          const val = params[0].value[1];
          const isSite = trueSites.includes(pos);
          return `Position: ${pos}<br/>Probability density: ${val.toFixed(6)}${isSite ? '<br/><b>True Site</b>' : ''}`;
        },
      },
      grid: {
        left: 60,
        right: 20,
        top: 30,
        bottom: 30,
      },
      xAxis: {
        type: 'value',
        name: 'Position',
        min: 0,
        max: Math.max(0, seqLength - 1),
        axisLabel: {
          color: MORANDI.textDark,
          fontSize: 10,
        },
        axisLine: { lineStyle: { color: '#ccc' } },
      },
      yAxis: {
        type: 'value',
        name: 'Density',
        min: 0,
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
          name: 'Probability density',
          data: x.map((position) => [position, density[position]]),
          symbol: 'none',
          lineStyle: { color, width: 2 },
          areaStyle: { color, opacity: 0.25 },
          markLine: {
            silent: true,
            symbol: ['none', 'none'],
            label: { show: false },
            lineStyle: {
              color: MORANDI.true_site,
              width: 1.5,
              type: 'dashed',
            },
            data: trueSites.map((position) => ({ xAxis: position })),
          },
        },
      ],
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
          {modelName}
        </span>
        <span style={{ fontSize: 11, color: '#888' }}>
          Gaussian KDE (σ={GAUSSIAN_BANDWIDTH}) · True sites: {trueSites.length}
        </span>
      </div>
      <div ref={chartRef} style={{ width: '100%', height: 150 }} />
    </div>
  );
};

const AttentionComparisonViz: React.FC = () => {
  const [selectedSample, setSelectedSample] = useState(0);
  const [sequenceIdInput, setSequenceIdInput] = useState('');
  const [requestedSequenceId, setRequestedSequenceId] = useState<number | undefined>();
  const [sequenceIdError, setSequenceIdError] = useState('');

  const { data, isLoading, isFetching, isError, error, refetch } = useQuery({
    queryKey: ['attentionComparison', requestedSequenceId],
    queryFn: () => fetchAttentionComparison(requestedSequenceId),
  });

  const loadNextBatch = async () => {
    setSequenceIdInput('');
    setSequenceIdError('');
    if (requestedSequenceId !== undefined) {
      setRequestedSequenceId(undefined);
      setSelectedSample(0);
      return;
    }
    const result = await refetch();
    if (result.data?.samples.length) {
      setSelectedSample(0);
    }
  };

  const loadSequenceById = () => {
    const value = sequenceIdInput.trim();
    if (!/^\d+$/.test(value)) {
      setSequenceIdError('请输入有效的数字序列 ID');
      return;
    }

    const sequenceId = Number(value);
    const availableSequenceIds = data?.available_sequence_ids;
    if (availableSequenceIds && !availableSequenceIds.includes(sequenceId)) {
      setSequenceIdError(`序列 ID ${sequenceId} 不在当前 NPZ 的 200 条预计算序列中`);
      return;
    }

    setSequenceIdError('');
    setRequestedSequenceId(sequenceId);
    setSelectedSample(0);
  };

  const sequenceSearch = (
    <div style={{
      display: 'flex',
      gap: 8,
      marginBottom: 16,
      alignItems: 'center',
    }}>
      <input
        type="text"
        inputMode="numeric"
        list="attention-sequence-ids"
        value={sequenceIdInput}
        onChange={(event) => {
          setSequenceIdInput(event.target.value);
          setSequenceIdError('');
        }}
        onKeyDown={(event) => {
          if (event.key === 'Enter') loadSequenceById();
        }}
        placeholder="输入序列 ID，按回车查询"
        aria-label="序列 ID"
        style={{
          width: 240,
          padding: '8px 12px',
          border: '1px solid #B8A9C9',
          borderRadius: 8,
          outline: 'none',
          color: MORANDI.textDark,
        }}
      />
      <datalist id="attention-sequence-ids">
        {data?.available_sequence_ids?.map((sequenceId) => (
          <option key={sequenceId} value={sequenceId} />
        ))}
      </datalist>
      {requestedSequenceId !== undefined && !isError && (
        <span style={{ fontSize: 12, color: '#666' }}>
          当前序列 ID：{requestedSequenceId}
        </span>
      )}
      {sequenceIdError && (
        <span style={{ fontSize: 12, color: '#d32f2f' }}>
          {sequenceIdError}
        </span>
      )}
    </div>
  );

  if (isLoading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400, color: '#666' }}>
        Loading attention comparison data...
      </div>
    );
  }

  if (isError) {
    return (
      <div>
        {sequenceSearch}
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 320, color: '#d32f2f' }}>
          Failed to load attention data: {error?.message}
        </div>
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
      {sequenceSearch}
      {/* Sample selector */}
      <div style={{
        display: 'flex',
        gap: 8,
        marginBottom: 16,
        flexWrap: 'wrap',
        alignItems: 'center',
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
        <button
          type="button"
          onClick={loadNextBatch}
          disabled={isFetching}
          style={{
            padding: '8px 18px',
            borderRadius: 8,
            border: '1px solid #8DA9C4',
            backgroundColor: isFetching ? '#E8E8E8' : '#8DA9C4',
            color: isFetching ? '#888' : '#fff',
            cursor: isFetching ? 'not-allowed' : 'pointer',
            fontWeight: 'bold',
            marginLeft: 4,
          }}
        >
          {isFetching ? '正在加载…' : '下一批'}
        </button>
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
