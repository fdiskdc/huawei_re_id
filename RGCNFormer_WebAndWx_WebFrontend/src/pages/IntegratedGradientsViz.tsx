import React, { useState, useEffect, useRef } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import { Spin, Alert, InputNumber, Select, Button, Space, Card, Typography } from 'antd';
import { Link } from 'react-router-dom';
import { useRna } from '../context/RnaContext';
import { useTranslation } from '../lib/i18n/LanguageContext';
import { fetchIntegratedGradients } from '../lib/api';

interface Node {
  id: string;
  label?: string;
  data?: {
    index: number;
    type: string;
    name: string;
    attributionScore?: number;
  };
  x?: number;
  y?: number;
  z?: number;
}

interface Link {
  source: string | Node;
  target: string | Node;
}

interface GraphData {
  nodes: Node[];
  edges: Link[];
}

const DESKTOP_BREAKPOINT = 768;
const SIDER_WIDTH_EXPANDED = 200;
const SIDER_WIDTH_COLLAPSED = 80;
const CONTENT_PADDING = 40;

const CLASS_NAMES = [
  'Am (0)', 'Atol (1)', 'Cm (2)', 'Gm (3)', 'Tm (4)', 'Y (5)',
  'ac4C (6)', 'm1A (7)', 'm5C (8)', 'm6A (9)', 'm6Am (10)', 'm7G (11)'
];

const IntegratedGradientsViz: React.FC = () => {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gcnData, setGcnData] = useState<GraphData | null>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [isDesktop, setIsDesktop] = useState(window.innerWidth > DESKTOP_BREAKPOINT);
  const [topN, setTopN] = useState<number>(10);
  const [targetClassId, setTargetClassId] = useState<number | null>(null);
  const [hasComputed, setHasComputed] = useState(false);
  
  const graphRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const resizeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { rnaSequence } = useRna();

  // Get top N node IDs for highlighting
  const topNodeIds = React.useMemo(() => {
    if (!gcnData || !hasComputed) return new Set<string>();

    // Sort nodes by absolute attribution score and get top N IDs
    const sortedNodes = [...gcnData.nodes]
      .sort((a: any, b: any) => Math.abs(b.data?.attributionScore || 0) - Math.abs(a.data?.attributionScore || 0))
      .slice(0, topN);
    
    return new Set(sortedNodes.map((node: Node) => node.id));
  }, [gcnData, hasComputed, topN]);

  // Update container size on window resize
  useEffect(() => {
    const updateSize = () => {
      if (!containerRef.current) return;

      const windowWidth = window.innerWidth;
      const windowHeight = window.innerHeight;
      const desktop = windowWidth > DESKTOP_BREAKPOINT;
      setIsDesktop(desktop);

      let availableWidth = windowWidth;
      if (desktop) {
        const siderElement = document.querySelector('.ant-layout-sider');
        const isCollapsed = siderElement?.classList.contains('ant-layout-sider-collapsed');
        const sidebarWidth = isCollapsed ? SIDER_WIDTH_COLLAPSED : SIDER_WIDTH_EXPANDED;
        availableWidth = windowWidth - sidebarWidth;
      }

      availableWidth -= CONTENT_PADDING;
      const availableHeight = windowHeight - CONTENT_PADDING;

      setContainerSize({ width: availableWidth, height: availableHeight });
    };

    updateSize();

    const handleResize = () => {
      if (resizeTimeoutRef.current) {
        clearTimeout(resizeTimeoutRef.current);
      }
      resizeTimeoutRef.current = setTimeout(updateSize, 100);
    };

    window.addEventListener('resize', handleResize);

    const observer = new MutationObserver(() => {
      updateSize();
    });

    const siderElement = document.querySelector('.ant-layout-sider');
    if (siderElement) {
      observer.observe(siderElement, {
        attributes: true,
        attributeFilter: ['class']
      });
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      if (resizeTimeoutRef.current) {
        clearTimeout(resizeTimeoutRef.current);
      }
      observer.disconnect();
    };
  }, [gcnData]);

  const handleCompute = async () => {
    if (!rnaSequence) {
      setError(t('No RNA sequence provided.'));
      return;
    }

    if (targetClassId === null) {
      setError(t('Please select a target class ID.'));
      return;
    }

    setLoading(true);
    setError(null);
    setHasComputed(false);

    try {
      const graphData: GraphData = await fetchIntegratedGradients({
        rnaSequence: rnaSequence,
        targetClassId: targetClassId
      });
      
      // Add labels to nodes and ensure position properties
      if (graphData.nodes) {
        graphData.nodes.forEach((node: any) => {
          node.label = node.id;
          // Ensure node has position properties for 3D rendering
          if (!node.x) node.x = 0;
          if (!node.y) node.y = 0;
          if (!node.z) node.z = 0;
        });
      }

      // Process links: convert string source/target to node references
      if (graphData.edges && graphData.nodes) {
        const nodeMap = new Map<string, Node>();
        graphData.nodes.forEach((node: Node) => {
          nodeMap.set(node.id, node);
        });

        graphData.edges.forEach((link: any) => {
          if (typeof link.source === 'string') {
            link.source = nodeMap.get(link.source);
          }
          if (typeof link.target === 'string') {
            link.target = nodeMap.get(link.target);
          }
        });
      }

      setGcnData(graphData);
      setHasComputed(true);
    } catch (e: any) {
      setError(t('Failed to compute Integrated Gradients: {message}').replace('{message}', e.message));
    } finally {
      setLoading(false);
    }
  };

  // Function to get node color based on attribution score
  const getNodeColor = (node: Node) => {
    const isTopNode = topNodeIds.has(node.id);

    // 1. 非关键节点：使用极浅的莫兰迪灰，几乎融入背景但保留轮廓
    if (!isTopNode) {
      return 'rgba(200, 207, 212, 0.3)';
    }

    // 2. 关键节点：采用深色调加强对比
    // 这里的逻辑是从“深莫兰迪蓝”过渡到“深碳灰色”
    const score = Math.abs(node.data?.attributionScore || 0);
    const normalized = Math.min(Math.max(score, 0), 1);

    // 颜色插值：从 深海蓝 (44, 62, 80) 到 炭黑 (28, 40, 51)
    const r = Math.floor(44 - (44 - 28) * normalized);
    const g = Math.floor(62 - (62 - 40) * normalized);
    const b = Math.floor(80 - (80 - 51) * normalized);

    return `rgb(${r}, ${g}, ${b})`; // 关键节点完全不透明，颜色深沉
  };

  if (!rnaSequence) {
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

  return (
    <div style={{ width: '100%' }}>
      {/* Introduction Card */}
      <Card style={{ marginBottom: 16, background: '#fff0f6', borderColor: '#eb2f96' }}>
        <Typography.Title level={4} style={{ color: '#eb2f96', margin: 0 }}>{t('Integrated Gradients')}</Typography.Title>
        <Typography.Paragraph style={{ marginTop: 8, marginBottom: 0 }}>
          {t('The Integrated Gradients visualization uses an explainable AI technique to attribute the model\'s predictions to specific nodes in the graph. It shows which nucleotides and features contribute most to the classification of a specific RNA modification type. Higher attribution scores indicate greater importance for the prediction.')}
        </Typography.Paragraph>
      </Card>

      <Space direction="vertical" style={{ width: '100%' }} size="large">
        {/* Control Panel */}
        <Card title={t('Integrated Gradients Controls')}>
          <Space wrap>
            <span>{t('Target Class ID:')}</span>
            <Select
              style={{ width: 200 }}
              placeholder={t('Select Target Class')}
              value={targetClassId}
              onChange={(value) => setTargetClassId(value)}
              options={CLASS_NAMES.map((name, index) => ({
                label: name,
                value: index,
              }))}
            />
            <span>{t('Show Top')}</span>
            <InputNumber 
              min={1} 
              max={1001} 
              value={topN} 
              onChange={(value) => setTopN(value || 1)} 
            />
            <span>{t('nodes')}</span>
            <Button type="primary" onClick={handleCompute} loading={loading}>
              {t('Compute')}
            </Button>
          </Space>
        </Card>

        {/* Graph Visualization */}
        <Card>
          {hasComputed && gcnData && (
            <div style={{ marginBottom: '16px', padding: '10px', background: '#f0f0f0', borderRadius: '4px' }}>
              <strong>{t('Statistics:')}</strong>
              <p>{t('Target Class:')} <strong>{CLASS_NAMES[targetClassId!]}</strong></p>
              <p>{t('Total Nodes:')} <strong>{gcnData.nodes.length}</strong></p>
              <p>{t('Total Edges:')} <strong>{gcnData.edges.length}</strong></p>
              <p>{t('Highlight:')} <strong>Top {topN}</strong>{t(' nodes (2x larger, opaque)')}</p>
              <p>{t('Other Nodes:')} <strong>{gcnData.nodes.length - topN}</strong>{t(' nodes (translucent)')}</p>
            </div>
          )}

          <div
            ref={containerRef}
            style={{
              width: '100%',
              height: isDesktop ? 'calc(100vh - 200px)' : 'calc(100vh - 300px)',
              position: 'relative',
              background: '#555963',
              borderRadius: '8px',
              overflow: 'hidden',
            }}
          >
            {loading && (
              <Spin
                tip={t('Computing Integrated Gradients...')}
                size="large"
                style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  zIndex: 100,
                }}
              />
            )}
            {error && (
              <Alert
                message={t('Error')}
                description={error}
                type="error"
                showIcon
                style={{ position: 'absolute', zIndex: 100, width: '80%', margin: '20px auto' }}
              />
            )}

            {gcnData && containerSize.width > 0 && containerSize.height > 0 && (
              <ForceGraph3D
                ref={graphRef}
                graphData={{
                  nodes: gcnData.nodes,
                  links: gcnData.edges
                }}
                width={containerSize.width}
                height={containerSize.height}
                nodeLabel="label"
                nodeColor={(node: any) => getNodeColor(node)}
                nodeVal={(node: any) => {
                  // 关键节点体积加大，形成重力感
                  return topNodeIds.has(node.id) ? 50000 : 1000;
                }}
                // 边：使用极细的浅灰绿线，保持图谱的整洁感
                linkColor={() => 'rgba(189, 195, 199, 0.4)'}
                linkWidth={20}
                linkOpacity={1}
                // 背景：经典的莫兰迪浅米白
                backgroundColor="#F8F9F9"
                enableNodeDrag={true}
                cooldownTicks={200}
              />
            )}
          </div>
        </Card>

        {/* Attribution Scores Table */}
        {hasComputed && gcnData && (
          <Card title={t('Node Attribution Scores (Top {topN})').replace('{topN}', String(topN))}>
            <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: '#f0f0f0', position: 'sticky', top: 0 }}>
                    <th style={{ padding: '8px', border: '1px solid #ddd', textAlign: 'left' }}>{t('Rank')}</th>
                    <th style={{ padding: '8px', border: '1px solid #ddd', textAlign: 'left' }}>{t('Node ID')}</th>
                    <th style={{ padding: '8px', border: '1px solid #ddd', textAlign: 'left' }}>{t('Type')}</th>
                    <th style={{ padding: '8px', border: '1px solid #ddd', textAlign: 'left' }}>{t('Position')}</th>
                    <th style={{ padding: '8px', border: '1px solid #ddd', textAlign: 'right' }}>{t('Attribution Score')}</th>
                  </tr>
                </thead>
                <tbody>
                  {gcnData.nodes
                    .filter((node: any) => node.data?.attributionScore !== undefined)
                    .sort((a: any, b: any) => Math.abs(b.data.attributionScore) - Math.abs(a.data.attributionScore))
                    .slice(0, topN)
                    .map((node: any, index: number) => (
                      <tr key={node.id}>
                        <td style={{ padding: '8px', border: '1px solid #ddd' }}>{index + 1}</td>
                        <td style={{ padding: '8px', border: '1px solid #ddd' }}>{node.id}</td>
                        <td style={{ padding: '8px', border: '1px solid #ddd' }}>{node.data.type}</td>
                        <td style={{ padding: '8px', border: '1px solid #ddd' }}>{node.data.index}</td>
                        <td style={{ padding: '8px', border: '1px solid #ddd', textAlign: 'right' }}>
                          {node.data.attributionScore?.toFixed(6)}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </Space>
    </div>
  );
};

export default IntegratedGradientsViz;
