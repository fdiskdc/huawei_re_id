/**
 * VizLayout.tsx - 可视化套件共用布局(桌面 Sider + 移动 Drawer) / Shared layout for viz suite
 *
 * App.tsx 中 <Route element={<VizLayout />}> 的目标组件,包裹 classification/attention/
 * gcn/target-gcn/integrated-gradients/model-viz 等可视化页面。
 * 根据视口宽度自动切换两套布局:
 *   - 桌面(>= 768px):左侧 Sider(导航菜单 + 返回首页 + 中英切换)+ 右侧 Content(标题栏 + Outlet)
 *   - 移动(< 768px):顶部 Header + 横向滚动菜单 + Content(Sider 自动折叠)
 * 配色采用莫兰迪色系(MORANDI 常量),整体淡雅。
 * The wrapper for visualization pages declared in App.tsx's <Route element={<VizLayout />}>.
 * Switches between two layouts based on viewport:
 *   - Desktop (>= 768px): left Sider (nav + return-home + i18n switch) + right Content (title + Outlet)
 *   - Mobile  (<  768px): top Header + horizontal scrollable menu + Content (Sider auto-collapsed)
 * Uses a Morandi-style color palette (MORANDI constant).
 *
 * 功能模块 / Modules:
 * - 响应式切换 (useEffect + window.innerWidth > DESKTOP_BREAKPOINT)
 *   / Responsive switch (useEffect + window resize listener)
 * - 桌面布局:固定 Sider + 滚动 Content / Desktop: fixed Sider + scrollable Content
 * - 移动布局:Header + 水平菜单 / Mobile: Header + horizontal menu
 * - i18n 切换按钮(中文/EN)/ i18n toggle buttons (zh / EN)
 * - 当前路由高亮(menuItemStyle 根据 currentKey)/ Active-route highlight
 *
 * 输入 / Inputs:
 * - window.innerWidth: 视口宽度 / Viewport width
 * - location.pathname: 当前路由 / Current route (from react-router useLocation)
 * - useTranslation().t / language / changeLanguage: i18n hook
 *
 * 输出 / Outputs:
 * - Ant Design <Layout>(Sider + Content)/ AntD <Layout> shell
 * - <Outlet /> 处挂载子路由 Page / <Outlet /> mounts child route page
 *
 * 数据流 / Data Flow:
 * 1. 用户访问 /classification 等 → App.tsx 匹配 VizLayout
 * 2. VizLayout 根据视口选 desktopLayout / mobileLayout
 * 3. 菜单点击 → navigate(key) → 路由更新 → 子 Page 重新挂载
 * 4. 中英按钮 → changeLanguage → LanguageContext 触发整树重渲染
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: lib/i18n/LanguageContext(useTranslation)
 * - 被调用 / Called by: App.tsx(<Route element={<VizLayout />}>)
 * - 包裹 / Wraps: pages/ClassificationViz, AttentionViz, GcnViz, TargetGcnViz,
 *   IntegratedGradientsViz, ModelViz
 *
 * 使用示例 / Usage Example:
 *   <Route element={<VizLayout />}>
 *     <Route path="/classification" element={<ClassificationViz />} />
 *     ...
 *   </Route>
 */
import React, { useState, useEffect } from 'react';
import { Layout, Button } from 'antd';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useTranslation } from '../lib/i18n/LanguageContext';

const { Content, Sider } = Layout;

const DESKTOP_BREAKPOINT = 768;

const MORANDI = {
  sidebarBg: '#9E9288',
  sidebarLogoBg: '#8B7E74',
  sidebarText: '#F0EBE6',
  sidebarActive: '#B8A9C9',
  sidebarActiveText: '#FFFFFF',
  contentBg: '#F0EBE6',
  titleColor: '#4A4A4A',
  titleUnderline: '#B8A9C9',
  btnBrown: '#C4A882',
};

const VizLayout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { t, changeLanguage, language } = useTranslation();
  const [isDesktop, setIsDesktop] = useState(window.innerWidth > DESKTOP_BREAKPOINT);
  const [siderCollapsed, setSiderCollapsed] = useState(false);

  const currentKey = location.pathname;

  useEffect(() => {
    const handleResize = () => {
      const desktop = window.innerWidth > DESKTOP_BREAKPOINT;
      setIsDesktop(desktop);
      if (!desktop) {
        setSiderCollapsed(true);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const pageTitles: Record<string, string> = {
    '/classic/viz/classification': t('Classification Results'),
    '/classic/viz/attention': t('Attention Weights'),
    '/classic/viz/gcn': t('GCN Graph Structure'),
    '/classic/viz/target-gcn': t('Target GCN'),
    '/classic/viz/integrated-gradients': t('Integrated Gradients'),
    '/classic/viz/model': t('Model Architecture'),
    '/classic/viz/reid': t('ReID Heatmap'),
  };

  const menuItems = [
    { key: '/classic/viz/classification', label: t('Classification Results') },
    { key: '/classic/viz/attention', label: t('Attention Weights') },
    { key: '/classic/viz/gcn', label: t('GCN Graph Structure') },
    { key: '/classic/viz/integrated-gradients', label: t('Integrated Gradients') },
    { key: '/classic/viz/model', label: t('Model Architecture') },
    { key: '/classic/viz/reid', label: t('ReID Heatmap') },
  ];

  const menuItemStyle = (key: string): React.CSSProperties => ({
    color: currentKey === key ? MORANDI.sidebarActiveText : MORANDI.sidebarText,
    backgroundColor: currentKey === key ? MORANDI.sidebarActive : 'transparent',
    borderRadius: 8,
    margin: '2px 8px',
    width: 'calc(100% - 16px)',
    paddingLeft: 16,
  });

  const desktopLayout = (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        collapsible
        collapsed={siderCollapsed}
        onCollapse={setSiderCollapsed}
        width={200}
        style={{
          overflow: 'auto',
          height: '100vh',
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: MORANDI.sidebarBg,
        }}
      >
        <div style={{ flexGrow: 1 }}>
          <div style={{
            height: 'auto',
            minHeight: 32,
            margin: 16,
            backgroundColor: MORANDI.sidebarLogoBg,
            borderRadius: 8,
            padding: '12px 8px',
            color: '#fff',
            fontWeight: 'bold',
            textAlign: 'center',
            wordBreak: 'break-word',
            overflow: 'hidden',
          }}>
            {siderCollapsed ? 'RNA' : t('RNA Visualization')}
          </div>
          <div style={{ paddingTop: 8 }}>
            {menuItems.map((item) => (
              <div
                key={item.key}
                onClick={() => navigate(item.key)}
                style={{
                  ...menuItemStyle(item.key),
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  padding: '10px 16px',
                  fontWeight: currentKey === item.key ? 600 : 400,
                }}
              >
                {item.label}
              </div>
            ))}
          </div>
        </div>
        <div style={{ padding: '16px', textAlign: 'center' }}>
          <Button
            type="primary"
            onClick={() => navigate('/classic')}
            style={{ marginBottom: '16px', width: '100%', backgroundColor: MORANDI.btnBrown, borderColor: MORANDI.btnBrown }}
          >
            {t('Return to Home')}
          </Button>
          <div>
            <Button
              onClick={() => changeLanguage('zh')}
              disabled={language === 'zh'}
              size="small"
              style={{ marginRight: 8 }}
            >
              中文
            </Button>
            <Button
              onClick={() => changeLanguage('en')}
              disabled={language === 'en'}
              size="small"
            >
              EN
            </Button>
          </div>
        </div>
      </Sider>
      <Layout style={{ marginLeft: siderCollapsed ? 80 : 200, transition: 'all 0.2s' }}>
        <Content style={{
          padding: 20,
          overflow: 'auto',
          minHeight: '100vh',
          backgroundColor: MORANDI.contentBg,
        }}>
          <h1 style={{
            fontSize: 22,
            fontWeight: 'bold',
            color: MORANDI.titleColor,
            marginBottom: 4,
          }}>
            {pageTitles[currentKey] || ''}
          </h1>
          <hr style={{
            border: 'none',
            borderTop: '2px solid #B8A9C9',
            marginBottom: 20,
          }} />
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );

  const mobileLayout = (
    <Layout style={{ minHeight: '100vh' }}>
      <div style={{
        backgroundColor: MORANDI.sidebarBg,
        padding: '12px 16px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div style={{
          color: '#fff',
          fontWeight: 'bold',
          fontSize: 14,
          wordBreak: 'break-word',
          maxWidth: '60%',
        }}>
          {t('RNA Visualization')}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <Button
            size="small"
            onClick={() => changeLanguage('zh')}
            disabled={language === 'zh'}
            style={{ color: language === 'zh' ? MORANDI.sidebarActive : '#fff', borderColor: 'transparent', background: 'transparent' }}
          >
            中文
          </Button>
          <Button
            size="small"
            onClick={() => changeLanguage('en')}
            disabled={language === 'en'}
            style={{ color: language === 'en' ? MORANDI.sidebarActive : '#fff', borderColor: 'transparent', background: 'transparent' }}
          >
            EN
          </Button>
          <Button size="small" type="primary" onClick={() => navigate('/classic')} style={{ backgroundColor: MORANDI.btnBrown, borderColor: MORANDI.btnBrown }}>
            {t('Return to Home')}
          </Button>
        </div>
      </div>
      <div style={{
        backgroundColor: MORANDI.sidebarBg,
        padding: '4px 0',
        overflowX: 'auto',
        whiteSpace: 'nowrap',
      }}>
        {menuItems.map((item) => (
          <div
            key={item.key}
            onClick={() => navigate(item.key)}
            style={{
              ...menuItemStyle(item.key),
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              padding: '8px 16px',
              margin: '4px 8px',
              fontWeight: currentKey === item.key ? 600 : 400,
              whiteSpace: 'nowrap',
            }}
          >
            {item.label}
          </div>
        ))}
      </div>
      <Content style={{
        padding: 16,
        overflow: 'auto',
        paddingBottom: 80,
        backgroundColor: MORANDI.contentBg,
      }}>
        <h1 style={{
          fontSize: 18,
          fontWeight: 'bold',
          color: MORANDI.titleColor,
          marginBottom: 4,
        }}>
          {pageTitles[currentKey] || ''}
        </h1>
        <hr style={{
          border: 'none',
          borderTop: '2px solid #B8A9C9',
          marginBottom: 16,
        }} />
        <Outlet />
      </Content>
    </Layout>
  );

  return isDesktop ? desktopLayout : mobileLayout;
};

export default VizLayout;
