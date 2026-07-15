import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import { Spin, Alert, Card, Typography } from 'antd';
import { Link } from 'react-router-dom';
import { useRna, type ClassificationResult } from '../context/RnaContext';
import { useTranslation } from '../lib/i18n/LanguageContext';
import { submitTask } from '../lib/api';

// 定义树节点类型
interface TreeNode {
  name: string;
  isPredicted?: boolean;
  children?: TreeNode[];
}

// 递归处理树数据，为每个节点添加样式
const processTreeData = (node: TreeNode): any => {
  const processed: any = {
    name: node.name,
    children: node.children?.map(child => processTreeData(child)),
  };

  // 根据 isPredicted 属性设置节点样式
  if (node.isPredicted) {
    // 预测结果：高亮显示（使用醒目的颜色和较大的节点）
    processed.itemStyle = {
      color: '#52c41a', // 绿色表示预测结果
      borderColor: '#389e0d',
      borderWidth: 2,
    };
    processed.label = {
      ...processed.label,
      color: '#52c41a',
      fontWeight: 'bold',
      fontSize: 14,
    };
    processed.symbolSize = 15; // 预测节点更大
  } else {
    // 未预测结果：灰色显示，但仍可见
    processed.itemStyle = {
      color: '#d9d9d9', // 灰色
      borderColor: '#bfbfbf',
      borderWidth: 1,
    };
    processed.label = {
      ...processed.label,
      color: '#8c8c8c',
      fontSize: 12,
    };
    processed.symbolSize = 10;
  }

  return processed;
};

interface ClassificationVizProps {
  data?: any;
}

const ClassificationViz: React.FC<ClassificationVizProps> = ({ data: propData }) => {
  const { t } = useTranslation();
  // 用于存储ECharts的配置对象
  const [options, setOptions] = useState({});
  // 用于控制加载状态，提供更好的用户体验
  const [loading, setLoading] = useState(true);
  // 用于存储请求过程中可能发生的错误
  const [error, setError] = useState<string | null>(null);

  const { rnaSequence, setClassificationResults } = useRna();

  // 使用useEffect在组件加载后执行数据获取操作
  useEffect(() => {
    const processDataAndSetResults = (treeData: any) => {
      const predictedNodes: ClassificationResult[] = [];
      function findPredicted(node: any) {
        if (node.isPredicted) {
          predictedNodes.push({ name: node.name, value: predictedNodes.length });
        }
        if (node.children) {
          node.children.forEach(findPredicted);
        }
      }
      findPredicted(treeData);
      setClassificationResults(predictedNodes);

      const processedData = processTreeData(treeData);
      setOptions({
        tooltip: {
          trigger: 'item',
          triggerOn: 'mousemove',
          formatter: (params: any) => {
            const isPredicted = params.data.itemStyle?.color === '#52c41a';
            const status = isPredicted ? t('✓ Predicted') : t('✗ Not predicted');
            return `<strong>${params.name}</strong><br/>${status}`;
          }
        },
        series: [
          {
            type: 'tree',
            data: [processedData],
            top: '5%',
            left: '10%',
            bottom: '5%',
            right: '10%',
            layout: 'orthogonal',
            orient: 'TB',
            symbolSize: 10,
            initialTreeDepth: 2,
            label: {
              position: 'top',
              distance: 5,
              fontSize: 12,
            },
            leaves: {
              label: {
                position: 'bottom',
                distance: 5,
              }
            },
            emphasis: {
              focus: 'descendant'
            },
            expandAndCollapse: true,
            animationDuration: 550,
            animationDurationUpdate: 750
          }
        ]
      });
      setLoading(false);
    };

    // If data is provided as prop, use it directly
    if (propData) {
      processDataAndSetResults(propData);
      return;
    }

    // Otherwise, fetch data from API using rnaSequence
    if (!rnaSequence) {
      setLoading(false);
      setError(t('No RNA sequence provided.'));
      return;
    }
    // 定义数据获取函数
    const fetchData = async () => {
      try {
        const apiData = await submitTask({
          jobId: crypto.randomUUID(),
          userId: 'user1',
          rnaSequence: rnaSequence
        });

        // 从API数据中提取分类结果部分
        const chartData = apiData.classification;

        processDataAndSetResults(chartData);

      } catch (e: any) {
        console.error("Failed to fetch or process data:", e);
        setError(t('Unable to load chart data: {message}').replace('{message}', e.message));
      } finally {
        // 无论成功或失败，都结束加载状态
        setLoading(false);
      }
    };

    fetchData();
      }, [propData, rnaSequence, setClassificationResults]);

  if (!propData && !rnaSequence) {
    return (
      <Alert
        message={t('Error')}
        description={
          <>
            {t('Please enter an RNA sequence.')} <Link to="/">{t('Return to Home')}</Link>
          </>
        }
        type="error"
        showIcon
      />
    );
  }

  // 根据加载和错误状态显示不同的UI
  if (loading) {
    return <Spin tip={t('Loading chart...')} size="large" style={{ display: 'block', marginTop: '50px' }} />;
  }

  if (error) {
    return <Alert message={t('Error')} description={error} type="error" showIcon />;
  }

  // 数据加载成功后，渲染图表
  return (
    <div style={{ width: '100%' }}>
      {/* Introduction Card */}
      <Card style={{ marginBottom: 16, background: '#f0f9ff', borderColor: '#1890ff' }}>
        <Typography.Title level={4} style={{ color: '#1890ff', margin: 0 }}>{t('Classification Tree')}</Typography.Title>
        <Typography.Paragraph style={{ marginTop: 8, marginBottom: 0 }}>
          {t('The Classification Tree visualization displays the hierarchical classification results for your RNA sequence. It shows the model\'s decision path from the root to the predicted RNA modifications. Predicted nodes are highlighted in green, while unselected paths are shown in gray. Use this visualization to understand how the model classifies your sequence and which modifications it predicts.')}
        </Typography.Paragraph>
      </Card>

      <div className="classification-viz-container" style={{ maxWidth: '100%', overflowX: 'auto' }}>
        <ReactECharts
          option={options}
          style={{ height: '80vh', width: '100%' }}
          notMerge={true}
          lazyUpdate={true}
        />
      </div>
    </div>
  );
};

export default ClassificationViz;
