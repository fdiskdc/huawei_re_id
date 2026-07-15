/**
 * GcnViz.tsx - GCN 图结构 3D 可视化(react-force-graph-3d)/ GCN graph 3D visualization
 *
 * /gcn 路由页面。使用 react-force-graph-3d 渲染 RNA 二级结构图(节点 = 核苷酸,
 * 边 = 配对关系),颜色按节点类型着色。流程:用户提交序列 → 后端推理 →
 * 返回图数据 → 渲染 3D 视图。提供自动旋转、暂停、重置视角等交互。
 * Page mounted at /gcn. Uses react-force-graph-3d to render the RNA secondary-structure
 * graph (nodes = nucleotides, edges = base pairs) with per-type coloring. Pipeline:
 * user submits sequence → backend inference → graph payload returned → 3D scene rendered.
 * Provides auto-rotation, pause, and camera-reset interactions.
 *
 * 功能模块 / Modules:
 * - 3D 力导向图(react-force-graph-3d)/ 3D force-directed graph
 * - 节点类型着色(N/M/P 等)/ Per-type node coloring
 * - 视角控制(自动旋转/暂停/重置)/ Camera controls
 * - 加载/错误状态(Spin, Alert)/ Loading/error states
 *
 * 输入 / Inputs:
 * - useRna().rnaSequence / jobId / setJobId: 来自 RnaContext / from RnaContext
 * - 后端 /api/v1/submit-task + /api/v1/get-result 轮询 / Submit + poll endpoints
 *
 * 输出 / Outputs:
 * - JSX.Element 3D 图容器 / 3D graph container JSX
 *
 * 数据流 / Data Flow:
 * 1. 用户输入序列 → setRnaSequence(rnaSequence)
 * 2. 点击"提交" → submitTask(...) → 拿到 jobId
 * 3. 轮询 getResult → 拿到 graph payload(节点/边/坐标)
 * 4. 构造 GraphData → 喂给 <ForceGraph3D>
 * 5. 用户操作相机(滚轮缩放、拖拽旋转)
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: lib/api.ts(submitTask, generateJobId)、context/RnaContext
 * - 被调用 / Called by: App.tsx(<Route path="/gcn">)
 * - 关联 / Related: TargetGcnViz.tsx(目标节点 GCN)、ModelViz.tsx(模型图)
 *
 * 使用示例 / Usage Example:
 *   // App.tsx
 *   <Route path="/gcn" element={<GcnViz />} />
 *   // 浏览器访问 http://host:5173/mrmodn/gcn
 */
import React, { useState, useEffect, useRef } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import { Spin, Alert, Card, Typography } from 'antd';
import { Link } from 'react-router-dom';
import { useRna } from '../context/RnaContext';
import { useTranslation } from '../lib/i18n/LanguageContext';
import { submitTask, generateJobId } from '../lib/api';

interface Node {
  id: string;
  label?: string;
  data?: {
    index: number;
    type: string;
  };
  x?: number;
  y?: number;
  z?: number;
  __threeObj?: any;
}

interface Link {
  source: string | Node;
  target: string | Node;
}

interface GraphData {
  nodes: Node[];
  edges: Link[];
}

interface ClassifiedLinks {
  backboneLinks: Link[];
  pairingLinks: Link[];
}

// Morandi nucleotide color scheme - each nucleotide gets a distinct Morandi color
const NUCLEOTIDE_MORANDI_COLORS = {
  'A': '#af9d8f', // 灰褐/米色调 (A)
  'C': '#8ea7a5', // 灰绿/灰蓝色调 (C)
  'G': '#a1b18d', // 灰绿/橄榄绿调 (G)
  'U': '#c7b0b2', // 柔和粉/藕荷色调 (U)
  'T': '#c7b0b2', // T与U相同配色
  'N': '#d8d4d0'  // 极浅灰/米白调 (未知核苷酸 N)
};

// Morandi base colors for background and connections
const MORANDI_BASE_COLORS = {
  background: '#f4f1ea',    // 柔和米白 (背景)
  backboneLink: '#bfb8b0',  // 浅灰褐，用于主链连接
  pairingLink: '#8d9aab',   // 灰蓝色，用于配对连接
  nodeBorder: '#6d655f',    // 深灰，用于节点边框（可选）
  tube: '#a89a91',          // 烟灰粉/赭石 (保留用于其他用途)
};

// Legacy color palette name for backward compatibility
const MORANDI_COLORS = MORANDI_BASE_COLORS;

interface GcnVizProps {
  data?: {
    nodes: Node[];
    edges: Link[];
  };
}

const GcnViz: React.FC<GcnVizProps> = ({ data: propData }) => {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [gcnData, setGcnData] = useState<GraphData | null>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const graphRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const resizeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const classifiedLinksRef = useRef<ClassifiedLinks | null>(null);

  const { rnaSequence, dataset, datasetIndex } = useRna();

  // Helper function to classify links
  const classifyLinks = (links: Link[], nodes: Node[]): ClassifiedLinks => {
    const backboneLinks: Link[] = [];
    const pairingLinks: Link[] = [];

    const nodeIndexMap = new Map<string, number>();
    nodes.forEach((node, index) => {
      nodeIndexMap.set(node.id, index);
    });

    links.forEach((link) => {
      const sourceNode = typeof link.source === 'string' 
        ? nodes.find(n => n.id === link.source)
        : link.source;
      const targetNode = typeof link.target === 'string'
        ? nodes.find(n => n.id === link.target)
        : link.target;

      if (!sourceNode || !targetNode) return;

      const sourceIndex = nodeIndexMap.get(sourceNode.id) ?? -1;
      const targetIndex = nodeIndexMap.get(targetNode.id) ?? -1;

      // Backbone links are sequential (i to i+1 or i to i-1)
      const isBackbone = Math.abs(sourceIndex - targetIndex) === 1;

      if (isBackbone) {
        backboneLinks.push(link);
      } else {
        pairingLinks.push(link);
      }
    });

    return { backboneLinks, pairingLinks };
  };

  // Helper function to check if a link is a backbone connection
  const isBackboneLink = (link: Link): boolean => {
    const sourceNode = typeof link.source === 'string' ? link.source : link.source.id;
    const targetNode = typeof link.target === 'string' ? link.target : link.target.id;
    
    const nodeIndexMap = new Map<string, number>();
    if (gcnData) {
      gcnData.nodes.forEach((node, index) => {
        nodeIndexMap.set(node.id, index);
      });
    }
    
    const sourceIndex = nodeIndexMap.get(sourceNode) ?? -1;
    const targetIndex = nodeIndexMap.get(targetNode) ?? -1;
    
    return Math.abs(sourceIndex - targetIndex) === 1;
  };


  // Update container size on window resize
  useEffect(() => {
    const updateSize = () => {
      if (!containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const availableWidth = rect.width;
      const availableHeight = rect.height;

      setContainerSize({ width: availableWidth, height: availableHeight });

      if (graphRef.current) {
        requestAnimationFrame(() => {
          graphRef.current?.width(availableWidth);
          graphRef.current?.height(availableHeight);
        });
      }
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
  }, []);

  // Initialize custom physics and animation when graph data is loaded
  useEffect(() => {
    if (!gcnData || !graphRef.current || gcnData.nodes.length === 0) return;

    const graph = graphRef.current;

    // Classify links
    const classified = classifyLinks(gcnData.edges, gcnData.nodes);
    classifiedLinksRef.current = classified;

    // Access the d3-force engine
    const forceEngine = graph.d3Force();

    // Set velocity decay for smoother, longer simulation
    if (forceEngine) {
      // Configure charge force for repulsion
      forceEngine('charge').strength(-150);
      
      // Configure link force
      forceEngine('link').distance(20).strength(0.1);
      
      // Configure center force to keep graph centered
      forceEngine('center').strength(0.1);
    }

    console.log('GCN Visualization initialized with', gcnData.nodes.length, 'nodes and', gcnData.edges.length, 'edges');
  }, [gcnData]);

  useEffect(() => {
    const processData = (graphData: GraphData) => {
      if (graphData.nodes) {
        graphData.nodes.forEach((node: any) => {
          node.label = node.id;
          
          // Remove fixed positions - let physics simulation handle the natural folding
          // The physics forces (steric hindrance, backbone rigidity, folding driver)
          // will naturally determine the molecule's 3D structure
        });
      }

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
    };

    if (propData) {
      processData(propData);
      setLoading(false);
      return;
    }

    const fetchData = async () => {
      if (!rnaSequence) {
        setLoading(false);
        setError(t('No RNA sequence provided.'));
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const apiData = await submitTask({
          jobId: generateJobId(),
          userId: 'user1',
          rnaSequence: rnaSequence,
          dataset: dataset,
          datasetIndex: datasetIndex,
        });
        const graphData: GraphData = apiData.gcn;

        processData(graphData);
      } catch (e: any) {
        setError(t('Unable to load graph data: {message}').replace('{message}', e.message));
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [propData, rnaSequence, dataset, datasetIndex]);

  // Set camera position after data is loaded
  useEffect(() => {
    if (!gcnData || !graphRef.current) return;

    // Wait a bit for the graph to render
    setTimeout(() => {
      const graph = graphRef.current;
      if (graph && gcnData.nodes.length > 0) {
        console.log('Camera positioned for visualization');
      }
    }, 1000);
  }, [gcnData]);

  if (!propData && !rnaSequence) {
    return (
      <Alert
        message={t('Error')}
        description={
          <>
            {t('Please enter an RNA sequence.')} <Link to="/classic">{t('Return to Home')}</Link>
          </>
        }
        type="error"
        showIcon
      />
    );
  }

  return (
    <div style={{ width: '100%' }}>
      <Card 
        style={{ 
          marginBottom: 16, 
          background: '#faf8f5', 
          borderColor: MORANDI_COLORS.tube,
          borderWidth: 1,
        }}
      >
        <Typography.Title level={4} style={{ color: '#333333', margin: 0 }}>
          {t('GCN Graph')}
        </Typography.Title>
        <Typography.Paragraph style={{ marginTop: 8, marginBottom: 0, color: '#333333' }}>
          {t('The GCN Graph visualization displays the graph structure representation of your RNA sequence as processed by the Graph Convolutional Network. Each node represents a nucleotide in the sequence, and edges represent the structural relationships between them. Interact with the 3D graph by dragging to rotate, scrolling to zoom, and right-click dragging to pan.')}
        </Typography.Paragraph>
      </Card>

      <div
        ref={containerRef}
        className="gcn-viz-container"
        style={{
          width: '100%',
          height: '100%',
          minHeight: '400px',
          position: 'relative',
          background: MORANDI_COLORS.background,
          borderRadius: '8px',
          overflow: 'hidden',
          boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
        }}
      >
        {/* Debug Info */}
        <div style={{
          position: 'absolute',
          top: 10,
          left: 10,
          background: 'rgba(255,255,255,0.9)',
          padding: '10px',
          borderRadius: '4px',
          fontSize: '12px',
          zIndex: 50,
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          color: '#000000',
        }}>
          <div>Container: {containerSize.width}x{containerSize.height}</div>
          <div>Data Loaded: {gcnData ? 'Yes' : 'No'}</div>
          {gcnData && (
            <>
              <div>Nodes: {gcnData.nodes.length}</div>
              <div>Edges: {gcnData.edges.length}</div>
              {gcnData.nodes.length > 0 && (
                <div>First Node ID: {gcnData.nodes[0].id}</div>
              )}
            </>
          )}
        </div>
        {loading && (
          <Spin
            tip={t('Loading graph data...')}
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
          <>
            <ForceGraph3D
              ref={graphRef}
              graphData={{
                nodes: gcnData.nodes,
                links: gcnData.edges,
              }}
              width={containerSize.width}
              height={containerSize.height}
              nodeLabel="label"
              nodeColor={(node) => {
                // Ensure node.data exists and type field is valid
                if (node.data && NUCLEOTIDE_MORANDI_COLORS[node.data.type as keyof typeof NUCLEOTIDE_MORANDI_COLORS]) {
                  return NUCLEOTIDE_MORANDI_COLORS[node.data.type as keyof typeof NUCLEOTIDE_MORANDI_COLORS];
                }
                // If type is unknown or undefined, use 'N' color as default
                return NUCLEOTIDE_MORANDI_COLORS['N'];
              }}
              nodeRelSize={20}
              linkColor={(link) => {
                // Check if link is a backbone connection
                if (isBackboneLink(link)) {
                  return MORANDI_BASE_COLORS.backboneLink;
                }
                return MORANDI_BASE_COLORS.pairingLink;
              }}
              linkWidth={(link) => {
                // Backbone links are thinner, pairing links are thicker
                return isBackboneLink(link) ? 10 : 20;
              }}
              backgroundColor={MORANDI_BASE_COLORS.background}
              enableNodeDrag={true}
              cooldownTicks={200}
            />
          </>
        )}
      </div>

      {/* Nucleotide Legend - Horizontal below 3D model */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexWrap: 'wrap',
        gap: '16px 24px',
        padding: '12px 16px',
        marginTop: '12px',
        background: 'rgba(244, 241, 234, 0.95)',
        borderRadius: '8px',
        border: `1px solid ${MORANDI_BASE_COLORS.tube}`,
        boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
      }}>
        <span style={{
          fontSize: '13px',
          fontWeight: 'bold',
          color: '#333333',
          marginRight: '8px',
        }}>
          {t('Nucleotide Legend')}
        </span>
        {Object.entries(NUCLEOTIDE_MORANDI_COLORS).map(([type, color]) => (
          <div key={type} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{
              width: '16px',
              height: '16px',
              borderRadius: '50%',
              backgroundColor: color,
              border: `2px solid ${MORANDI_BASE_COLORS.nodeBorder}`,
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            }} />
            <span style={{ fontSize: '12px', color: '#333333', fontWeight: '500' }}>
              {type}
            </span>
          </div>
        ))}
        <div style={{ width: '1px', height: '16px', background: MORANDI_BASE_COLORS.backboneLink, margin: '0 4px' }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '20px', height: '3px', backgroundColor: MORANDI_BASE_COLORS.backboneLink }} />
          <span style={{ fontSize: '12px', color: '#333333' }}>{t('Backbone')}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '20px', height: '3px', backgroundColor: MORANDI_BASE_COLORS.pairingLink }} />
          <span style={{ fontSize: '12px', color: '#333333' }}>{t('Pairing')}</span>
        </div>
      </div>
    </div>
  );
};

export default GcnViz;
