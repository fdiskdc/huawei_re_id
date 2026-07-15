/*
 * @Author: Chao Deng && chaodeng987@outlook.com
 * @Date: 2026-01-21 16:52:03
 * @LastEditors: Chao Deng && chaodeng987@outlook.com
 * @LastEditTime: 2026-01-21 17:25:57
 * @FilePath: /rgcnformer_mobile_web/frontend/src/lib/i18n/LanguageContext.tsx
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
