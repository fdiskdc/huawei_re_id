/**
 * main.tsx - React 应用入口与全局 Provider 装配 / React app entry & global provider assembly
 *
 * 浏览器入口:创建 React 根、加载全局样式、装配 BrowserRouter、RnaProvider、LanguageProvider
 * 三个全局 Provider,然后渲染 <App /> 路由树。所有可视化页面都通过此入口挂载。
 * 部署在站点根路径 `/`,域名由 Nginx 配置决定。
 * Browser entry: creates the React root, loads global styles, and composes three global
 * providers (BrowserRouter → RnaProvider → LanguageProvider) before rendering the <App />
 * route tree. All visualization pages are mounted through this entry. The app is served
 * at the site root; the public domain is configured by Nginx.
 *
 * 功能模块 / Modules:
 * - createRoot(...).render(): React 18 根节点创建 + 渲染 / React 18 root + render
 * - BrowserRouter: HTML5 History 根路径路由 / HTML5 history router at site root
 * - RnaProvider: RNA 全局状态(序列、模型结果、对比数据)/ RNA global state context
 * - LanguageProvider: 中英双语运行时切换 / Bilingual i18n runtime switch
 * - StrictMode: 开发期双调用、过期 API 警告 / Dev double-invoke + deprecated API warning
 *
 * 输入 / Inputs:
 * - document.getElementById('root'): HTML 根挂载点 / HTML root mount node
 * - window.location: 浏览器 URL(Vite 启动时 base 决定 prefix)/ Browser URL
 *
 * 输出 / Outputs:
 * - 挂载在 #root 上的 React 应用实例 / React app instance mounted on #root
 *
 * 数据流 / Data Flow:
 * 1. main.tsx 加载 → createRoot 拿到 DOM 节点
 * 2. 装配 Provider 栈(BrowserRouter → RnaProvider → LanguageProvider)
 * 3. 渲染 <App /> 路由树
 * 4. 各 Page 组件按 URL 匹配并加载
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: App.tsx(根组件 / root component)
 * - 依赖 / Depends on: context/RnaContext.tsx, lib/i18n/LanguageContext.tsx, index.css
 * - 部署 / Served by: index.html(<div id="root">) + Vite 构建 / Vite build
 *
 * 使用示例 / Usage Example:
 *   // index.html
 *   <div id="root"></div>
 *   <script type="module" src="/src/main.tsx"></script>
 *   // Vite 启动: npm run dev → http://localhost:5173/
 */
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { RnaProvider } from './context/RnaContext'
import { LanguageProvider } from './lib/i18n/LanguageContext'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter basename="/mrmodn">
      <RnaProvider>
        <LanguageProvider>
          <App />
        </LanguageProvider>
      </RnaProvider>
    </BrowserRouter>
  </StrictMode>,
)
