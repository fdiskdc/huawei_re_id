/**
 * LanguageContext.tsx - 中英双语运行时切换 Context / Bilingual i18n context
 *
 * React Context,提供整树范围内中英双语运行时切换能力:
 *   - language: 当前语言('en' | 'zh')
 *   - changeLanguage(lang): 切换语言,触发整树重渲染
 *   - t(key): 查表函数,从 resources[language] 中取值
 * 文案资源来自 ./en.ts 与 ./zh.ts(结构对称,key 完全相同)。
 * React Context for tree-wide bilingual i18n at runtime:
 *   - language: current language ('en' | 'zh')
 *   - changeLanguage(lang): switch language, triggers tree re-render
 *   - t(key): lookup function that resolves a key from resources[language]
 * String resources come from ./en.ts and ./zh.ts (structurally symmetric, same key set).
 *
 * 功能模块 / Modules:
 * - Language 类型('en' | 'zh')/ Language type
 * - resources: { en, zh } 文案资源 / Resources map
 * - LanguageProvider: Context.Provider + useState(持久化 language)
 *   / Context provider with internal state
 * - useTranslation(): 自定义 hook,封装 useContext / useTranslation() custom hook
 *
 * 输入 / Inputs:
 * - children: ReactNode(由 main.tsx 与 App.tsx 双重包裹)
 *   / React children wrapped twice (main.tsx and App.tsx)
 * - 用户点击 VizLayout 的中英按钮 → changeLanguage / Triggered by i18n buttons in VizLayout
 *
 * 输出 / Outputs:
 * - { language, changeLanguage, t } 暴露给消费组件 / Exposed to consumers
 *
 * 数据流 / Data Flow:
 * 1. main.tsx 装配 <LanguageProvider><App /></LanguageProvider>
 * 2. App.tsx 内层又包了一层 <LanguageProvider>(保证独立运行时切换)
 * 3. 任何组件调 const { t, changeLanguage } = useTranslation() 获取翻译函数
 * 4. 用户点击"中文"/"EN" → changeLanguage('zh'|'en') → Provider state 更新 → 整树重渲染
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: lib/i18n/en.ts, lib/i18n/zh.ts(资源)
 * - 被调用 / Called by: main.tsx, App.tsx(双重 Provider)、所有 Page(用 useTranslation)
 * - 关联 / Related: components/VizLayout.tsx(i18n 切换按钮)
 *
 * 使用示例 / Usage Example:
 *   // VizLayout.tsx
 *   const { t, changeLanguage, language } = useTranslation();
 *   <Button onClick={() => changeLanguage('zh')}>中文</Button>
 *   <h1>{t('Classification Results')}</h1>
 */
/*
 * @Author: Chao Deng && chaodeng987@outlook.com
 * @Date: 2026-01-21 16:52:03
 * @LastEditors: Chao Deng && chaodeng987@outlook.com
 * @LastEditTime: 2026-01-21 17:25:57
 * @FilePath: /mrmodn_mobile_web/frontend/src/lib/i18n/LanguageContext.tsx
 * @Description: 
 * 那只是一场游戏一场梦
 *  
 * https://orcid.org/0009-0009-8520-1656
 * DOI: 10.3390/app15158626
 * DOI: 10.3390/rs17142354
 * Copyright (c) 2026 by ${Chao Deng}, All Rights Reserved. 
 */
import { createContext, useState, useContext, type ReactNode } from 'react';
import { en } from './en';
import { zh } from './zh';

type Language = 'en' | 'zh';

interface LanguageContextType {
  language: Language;
  changeLanguage: (lang: Language) => void;
  t: (key: string) => string;
}

const resources: { [key in Language]: typeof en | typeof zh } = {
  en,
  zh,
};

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const LanguageProvider = ({ children }: { children: ReactNode }) => {
  const [language, setLanguage] = useState<Language>('en');

  const changeLanguage = (lang: Language) => {
    setLanguage(lang);
  };

  const t = (key: string) => {
    const langResources = resources[language];
    return langResources.translation[key as keyof typeof langResources.translation] || key;
  };

  return (
    <LanguageContext.Provider value={{ language, changeLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useTranslation = () => {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useTranslation must be used within a LanguageProvider');
  }
  return context;
};
