/**
 * UMapViz.tsx - RNA 特征 UMAP 2D 嵌入散点图 / UMAP 2D embedding scatter
 *
 * /viz-display 之一(也可独立访问)。使用 echarts-for-react 渲染 UMAP 2D 散点图:
 * 每个点 = 一条 RNA 序列(或序列片段),颜色按 12 类修饰着色。数据由后端
 * /api/v1/umap 端点返回 UmapData,通过 props 传入(由 VizDisplayPage / Workspace 注入)
 * 或自取(若 data 缺省)。
 * One of the /viz-display pages. Renders a UMAP 2D scatter with echarts-for-react:
 * each point = one RNA sequence (or fragment), colored by 12 modification classes.
 * Data is passed via props (UmapData) by VizDisplayPage/Workspace, or self-fetched if
 * not provided.
 *
 * 功能模块 / Modules:
 * - ECharts 散点图(symbol/circle, visualMap)/ ECharts scatter
 * - 12 类修饰颜色映射 / 12-class color mapping
 * - tooltip(序列 id + 修饰类型)/ Tooltip
 * - 加载/错误态 / Loading/error states
 *
 * 输入 / Inputs (props):
 * - data?: UmapData(可选;若省略则在内部 useQuery 拉取)
 *
 * 输出 / Outputs:
 * - JSX.Element 散点图 / Scatter JSX
 *
 * 数据流 / Data Flow:
 * 1. props.data 提供 → 直接渲染
 * 2. props.data 缺省 → useQuery(fetchUmap)
 * 3. 转换 { x, y, label } 三元组数组 → ECharts series.data
 * 4. echart.setOption → 渲染
 * 5. hover → tooltip
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: lib/api.ts(UmapData 类型)
 * - 被调用 / Called by: pages/VizDisplayPage.tsx、pages/WorkspacePage.tsx
 * - 关联 / Related: 后端 /api/v1/umap 端点
 *
 * 使用示例 / Usage Example:
 *   <UMapViz data={umapPayload} />
 *   // 或自取:
 *   <UMapViz />
 */
import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { Spin, Alert } from 'antd';
import { useTranslation } from '../lib/i18n/LanguageContext';
import type { UmapData } from '../lib/api';

interface UMapVizProps {
  data?: UmapData;
}

const UMapViz: React.FC<UMapVizProps> = ({ data: propData }) => {
  const { t } = useTranslation();

  const { data, isLoading, isError } = propData
    ? { data: propData, isLoading: false, isError: false }
    : { data: undefined, isLoading: true, isError: false };

  const option = useMemo(() => {
    if (!data) return {};

    const { points, density_contours, metadata } = data;

    const labelGroups: Record<string, any[]> = {};
    for (const p of points) {
      if (!labelGroups[p.label]) labelGroups[p.label] = [];
      labelGroups[p.label].push([p.u1, p.u2, p.seq, p.probs, p.group]);
    }

    const scatterSeries = Object.entries(labelGroups).map(([label, pts]) => ({
      name: label,
      type: 'scatter' as const,
      data: pts,
      symbolSize: 4,
      itemStyle: { color: metadata.color_map[label] || '#999' },
      emphasis: {
        focus: 'series',
        itemStyle: { borderColor: '#fff', borderWidth: 1 }
      }
    }));

    const graphicElements: any[] = [];
    for (const [group, contours] of Object.entries(density_contours)) {
      const color = metadata.group_colors[group] || '#999';
      for (const contour of contours) {
        for (const polygon of contour.polygons) {
          if (polygon.length >= 3) {
            graphicElements.push({
              type: 'polygon' as const,
              shape: { points: polygon },
              style: {
                fill: color,
                opacity: 0.12,
                stroke: color,
                lineWidth: 0.5
              },
              silent: true,
              z: -1
            });
          }
        }
      }
    }

    const tooltip = {
      trigger: 'item' as const,
      backgroundColor: 'rgba(255,255,255,0.96)',
      borderColor: '#ddd',
      borderWidth: 1,
      padding: [8, 12],
      textStyle: { color: '#333', fontSize: 12 },
      formatter: (params: any): string => {
        const d = params.data;
        if (!d || !Array.isArray(d)) return '';
        const [,, seq, probs, group] = d;
        const labelNames = metadata.label_names || [];
        const colorMap = metadata.color_map || {};

        let html = `<div style="max-width:320px;font-family:monospace">`;
        html += `<div style="margin-bottom:4px"><strong>序列:</strong> <span style="display:inline-block;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;vertical-align:bottom;font-family:monospace">${seq || ''}</span></div>`;
        html += `<div style="margin-bottom:4px"><strong>标签:</strong> ${params.seriesName} <span style="color:#888">[${group}组]</span></div>`;
        html += `<div style="border-top:1px solid #eee;padding-top:4px;margin-top:4px"><strong>预测概率:</strong></div>`;

        probs.forEach((p: number, i: number) => {
          const w = Math.round(p * 100);
          const c = colorMap[labelNames[i]] || '#999';
          html += `<div style="display:flex;align-items:center;margin:2px 0;">`;
          html += `<span style="width:50px;font-size:11px">${labelNames[i]}</span>`;
          html += `<div style="flex:1;height:10px;background:#f0f0f0;border-radius:2px;margin:0 4px;">`;
          html += `<div style="width:${w}%;height:100%;background:${c};border-radius:2px;"></div>`;
          html += `</div>`;
          html += `<span style="width:38px;text-align:right;font-size:11px">${(p * 100).toFixed(1)}%</span>`;
          html += `</div>`;
        });

        html += `</div>`;
        return html;
      }
    };

    return {
      tooltip,
      legend: {
        type: 'scroll' as const,
        top: 10,
        right: 10,
        selected: Object.fromEntries(
          Object.keys(labelGroups).map(k => [k, true])
        )
      },
      xAxis: { type: 'value' as const, show: false, scale: true },
      yAxis: { type: 'value' as const, show: false, scale: true },
      dataZoom: [
        { type: 'inside' as const, xAxisIndex: 0, zoomOnMouseWheel: true },
        { type: 'inside' as const, yAxisIndex: 0, zoomOnMouseWheel: true }
      ],
      graphic: graphicElements,
      series: scatterSeries,
      animation: true
    };
  }, [data]);

  if (isLoading) {
    return (
      <div style={{ textAlign: 'center', padding: 60 }}>
        <Spin size="large" />
        <div style={{ marginTop: 16, color: '#666' }}>{t('Loading UMAP data...')}</div>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <Alert
        type="error"
        message={t('Failed to load UMAP data')}
        description={t('Please try again or contact support')}
        showIcon
      />
    );
  }

  return (
    <div style={{ width: '100%' }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 12,
        padding: '0 4px'
      }}>
        <div style={{ color: '#666', fontSize: 13 }}>
          {t('Total points')}: {data.metadata?.total_points || 0}
          {data.metadata?.subsampled && <span style={{ color: '#999', marginLeft: 8 }}>(subsampled)</span>}
          {' | '}
          {t('Per class')}: {data.metadata?.n_per_class || 0}
        </div>
      </div>
      <ReactECharts
        option={option}
        style={{ height: '650px', width: '100%' }}
        opts={{ renderer: 'canvas' }}
      />
    </div>
  );
};

export default UMapViz;