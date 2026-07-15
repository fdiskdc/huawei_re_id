/**
 * AttentionDistributionViz.tsx - 注意力分布直方图 / Attention distribution histogram
 *
 * /viz-display 之一(也可独立访问)。使用 ECharts 渲染 Transformer 注意力权重
 * 的整体分布(直方图 + KDE 曲线),x 轴 = 注意力值(0-1),y 轴 = 频次。
 * 数据由 fetchAttentionVisualization 拉取,经 useQuery 缓存。
 * One of the /viz-display pages. Renders an ECharts histogram (with optional KDE curve)
 * of the global attention weight distribution: x-axis = attention value (0-1), y-axis =
 * frequency. Data via fetchAttentionVisualization + useQuery.
 *
 * 功能模块 / Modules:
 * - ECharts 直方图(bar) + 折线(KDE)/ ECharts bar + line
 * - 莫兰迪色卡 / Morandi palette
 * - 加载/错误态 / Loading/error states
 *
 * 输入 / Inputs:
 * - useRna().rnaSequence: 当前序列(可影响数据)
 * - 后端 /api/v1/attention-visualization 返回 attention 分布
 *
 * 输出 / Outputs:
 * - JSX.Element 直方图 / Histogram JSX
 *
 * 数据流 / Data Flow:
 * 1. useQuery → fetchAttentionVisualization(seq?)
 * 2. 转换为 ECharts bins 数组
 * 3. echart.setOption({ series: [bar, line] })
 * 4. hover → tooltip
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: lib/api.ts(fetchAttentionVisualization)、context/RnaContext
 * - 被调用 / Called by: App.tsx(<Route path="/viz-display"> 内部)
 * - 关联 / Related: AttentionViz.tsx(位置级)、AttentionComparisonViz.tsx(多模型)
 *
 * 使用示例 / Usage Example:
 *   const { data } = useQuery({
 *     queryKey: ['attention-distribution', rnaSequence],
 *     queryFn: () => fetchAttentionVisualization(),
 *   });
 */
import React from 'react';
import { useQuery } from '@tanstack/react-query';
import * as echarts from 'echarts';
import { useRna } from '../context/RnaContext';
import { useTranslation } from '../lib/i18n/LanguageContext';
import { fetchAttentionVisualization } from '../lib/api';

const MORANDI = {
  bg: '#FAFAF8',
  textDark: '#4A4A4A',
  curve: '#84A59D',
  threshold: 0.5,
};

const MODIFICATION_COLORS: Record<string, string> = {
  'Am': '#8DA9C4',
  'Atol': '#B5838D',
  'Cm': '#A3B18A',
  'Gm': '#DDB892',
  'Tm': '#6B705C',
  'Y': '#9E9E9E',
  'ac4C': '#C9ADA7',
  'm1A': '#A28B8B',
  'm5C': '#8B9DAF',
  'm6A': '#B8A9C9',
  'm6Am': '#C4A882',
  'm7G': '#D4A0A0',
};

interface AttentionChartProps {
  attention: number[];
  className: string;
  probability: number;
  seqLength: number;
}

const AttentionChart: React.FC<AttentionChartProps> = ({
  attention,
  className,
  probability,
  seqLength,
}) => {
  const chartRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!chartRef.current) return;

    const chart = echarts.init(chartRef.current);
    const x = Array.from({ length: seqLength }, (_, i) => i);
    const color = MODIFICATION_COLORS[className] || '#888';

    const option: echarts.EChartsOption = {
      backgroundColor: MORANDI.bg,
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const pos = params[0].dataIndex;
          const val = params[0].value;
          return `Position: ${pos}<br/>Attention: ${val.toFixed(6)}`;
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
    };

    chart.setOption(option);

    const handleResize = () => chart.resize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.dispose();
    };
  }, [attention, className, seqLength]);

  return (
    <div style={{
      marginBottom: 16,
      padding: 16,
      backgroundColor: '#fff',
      borderRadius: 12,
      boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 8,
      }}>
        <h3 style={{
          margin: 0,
          fontSize: 16,
          fontWeight: 'bold',
          color: MODIFICATION_COLORS[className] || MORANDI.textDark,
        }}>
          {className}
        </h3>
        <span style={{
          fontSize: 13,
          color: probability > MORANDI.threshold ? '#27AE60' : '#E74C3C',
          fontWeight: 'bold',
        }}>
          P = {(probability * 100).toFixed(1)}%
        </span>
      </div>
      <div ref={chartRef} style={{ width: '100%', height: 200 }} />
    </div>
  );
};

const AttentionDistributionViz: React.FC = () => {
  const { t } = useTranslation();
  const { rnaSequence } = useRna();

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['attentionVisualization', rnaSequence],
    queryFn: () => fetchAttentionVisualization({ rnaSequence: rnaSequence! }),
    enabled: !!rnaSequence,
  });

  if (!rnaSequence) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: 400,
        fontSize: 16,
        color: '#666',
      }}>
        {t('No RNA sequence provided.')}
      </div>
    );
  }

  if (isLoading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: 400,
        fontSize: 16,
        color: '#666',
      }}>
        {t('Loading attention visualization...')}
      </div>
    );
  }

  if (isError) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: 400,
        fontSize: 16,
        color: '#d32f2f',
      }}>
        {t('Failed to load attention data: {message}').replace('{message}', error?.message || '')}
      </div>
    );
  }

  if (!data) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: 400,
        fontSize: 16,
        color: '#666',
      }}>
        {t('No attention data available.')}
      </div>
    );
  }

  // Filter classes with probability > threshold
  const activeClasses = data.classes.filter((c) => c.probability > MORANDI.threshold);

  if (activeClasses.length === 0) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: 400,
        fontSize: 16,
        color: '#666',
      }}>
        {t('No modifications detected with probability > 50%.')}
      </div>
    );
  }

  return (
    <div>
      {/* Info banner */}
      <div style={{
        marginBottom: 16,
        padding: '12px 16px',
        backgroundColor: '#F5F0EB',
        borderRadius: 8,
        fontSize: 13,
        color: MORANDI.textDark,
      }}>
        <b>Attention Distribution</b> — Showing {activeClasses.length} modifications with probability &gt; 50%
      </div>

      {/* Attention charts */}
      {activeClasses.map((cls) => (
        <AttentionChart
          key={cls.index}
          attention={cls.attention}
          className={cls.name}
          probability={cls.probability}
          seqLength={data.sequence_length}
        />
      ))}
    </div>
  );
};

export default AttentionDistributionViz;
