import React, { useState, useEffect } from 'react';
import { Layout, Menu, Button } from 'antd';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useTranslation } from '../lib/i18n/LanguageContext';

const { Content, Footer, Sider } = Layout;

const DESKTOP_BREAKPOINT = 768;

const VizLayout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { t, changeLanguage, language } = useTranslation();
  const [isDesktop, setIsDesktop] = useState(window.innerWidth > DESKTOP_BREAKPOINT);
  const [siderCollapsed, setSiderCollapsed] = useState(false);

  const currentKey = location.pathname;

  const handleMenuClick = (e: any) => {
    navigate(e.key);
  };

  useEffect(() => {
    const handleResize = () => {
      const desktop = window.innerWidth > DESKTOP_BREAKPOINT;
      setIsDesktop(desktop);
      // Auto-collapse sidebar on mobile
      if (!desktop) {
        setSiderCollapsed(true);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const menuItems = [
    { key: '/classification', label: t('Classification Results') },
    { key: '/attention', label: t('Attention Weights') },
    { key: '/gcn', label: t('GCN Graph Structure') },
    { key: '/integrated-gradients', label: t('Integrated Gradients') },
    { key: '/model-viz', label: t('Model Architecture') },
    { key: '/reid', label: 'reid' },
  ];

  // Desktop layout: sidebar on the left
  if (isDesktop) {
    return (
      <Layout style={{ minHeight: '100vh' }}>
        <Sider
          collapsible
          collapsed={siderCollapsed}
          onCollapse={setSiderCollapsed}
          theme="dark"
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
          }}
        >
          <div style={{ flexGrow: 1 }}>
            <div style={{
              height: 32,
              margin: 16,
              color: '#fff',
              fontWeight: 'bold',
              textAlign: 'center',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
            }}>
              {siderCollapsed ? 'RNA' : t('RNA Visualization')}
            </div>
            <Menu
              onClick={handleMenuClick}
              selectedKeys={[currentKey]}
              mode="inline"
              theme="dark"
              items={menuItems}
              inlineIndent={16}
            />
          </div>
          <div style={{ padding: '16px', textAlign: 'center' }}>
            <Button
              type="primary"
              onClick={() => navigate('/')}
              style={{ marginBottom: '16px', width: '100%' }}
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
            padding: '20px',
            overflow: 'auto',
            minHeight: '100vh',
          }}>
            <Outlet />
          </Content>
        </Layout>
      </Layout>
    );
  }

  // Mobile layout: menu at the bottom
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Content style={{
        padding: '20px',
        overflow: 'auto',
        paddingBottom: 60, // Space for footer
      }}>
        <Outlet />
      </Content>
      <Footer style={{ padding: 0, position: 'fixed', bottom: 0, left: 0, right: 0, zIndex: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '0 8px', backgroundColor: '#001529' }}>
            <Button
              onClick={() => navigate('/')}
              size="small"
              ghost
              style={{ marginRight: 16, color: 'rgba(255, 255, 255, 0.65)' }}
            >
              {t('Return to Home')}
            </Button>
            <Button
              onClick={() => changeLanguage('zh')}
              disabled={language === 'zh'}
              size="small"
              ghost
              style={{ marginRight: 8, color: language === 'zh' ? '#1890ff' : 'rgba(255, 255, 255, 0.65)' }}
            >
              中文
            </Button>
            <Button
              onClick={() => changeLanguage('en')}
              disabled={language === 'en'}
              size="small"
              ghost
              style={{ color: language === 'en' ? '#1890ff' : 'rgba(255, 255, 255, 0.65)' }}
            >
              EN
            </Button>
        </div>
        <Menu
          onClick={handleMenuClick}
          selectedKeys={[currentKey]}
          mode="horizontal"
          theme="dark"
          style={{ justifyContent: 'center' }}
          items={menuItems}
        />
      </Footer>
    </Layout>
  );
};

export default VizLayout;
