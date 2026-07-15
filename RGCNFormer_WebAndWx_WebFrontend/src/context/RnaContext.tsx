/*
 * @Author: Chao Deng && chaodeng987@outlook.com
 * @Date: 2026-01-20 12:06:34
 * @LastEditors: Chao Deng && chaodeng987@outlook.com
 * @LastEditTime: 2026-01-20 16:45:57
 * @FilePath: /rgcnformer_mobile_web/frontend/src/context/RnaContext.tsx
 * @Description: 
 * 那只是一场游戏一场梦
 *  
 * https://orcid.org/0009-0009-8520-1656
 * DOI: 10.3390/app15158626
 * DOI: 10.3390/rs17142354
 * Copyright (c) 2026 by ${Chao Deng}, All Rights Reserved. 
 */
import React, { createContext, useState, useContext, type ReactNode } from 'react';

export interface ClassificationResult {
    name: string;
    value: number;
}

interface RnaContextType {
    rnaSequence: string;
    setRnaSequence: (sequence: string) => void;
    server: string;
    setServer: (server: string) => void;
    classificationResults: ClassificationResult[];
    setClassificationResults: (results: ClassificationResult[]) => void;
}

const RnaContext = createContext<RnaContextType | undefined>(undefined);

export const RnaProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [rnaSequence, setRnaSequence] = useState('');
    const [server, setServer] = useState('');
    const [classificationResults, setClassificationResults] = useState<ClassificationResult[]>([]);

    return (
        <RnaContext.Provider value={{ rnaSequence, setRnaSequence, server, setServer, classificationResults, setClassificationResults }}>
            {children}
        </RnaContext.Provider>
    );
};

export const useRna = () => {
    const context = useContext(RnaContext);
    if (!context) {
        throw new Error('useRna must be used within a RnaProvider');
    }
    return context;
};
