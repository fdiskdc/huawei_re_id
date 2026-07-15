/*
 * @Author: Chao Deng && chaodeng987@outlook.com
 * @Date: 2026-01-20 08:39:31
 * @LastEditors: Chao Deng && chaodeng987@outlook.com
 * @LastEditTime: 2026-01-22 21:10:44
 * @FilePath: /rgcnformer_mobile_web/frontend/src/pages/MainPage.tsx
 * @Description: 
 * 那只是一场游戏一场梦
 *  
 * https://orcid.org/0009-0009-8520-1656
 * DOI: 10.3390/app15158626
 * DOI: 10.3390/rs17142354
 * Copyright (c) 2026 by ${Chao Deng}, All Rights Reserved. 
 */
import React, { useState } from 'react';
import { Input, Select, Button, Card, message, Space } from 'antd';
import { useNavigate } from 'react-router-dom';
import { useRna } from '../context/RnaContext';
import { useTranslation } from '../lib/i18n/LanguageContext';
import { submitTask } from '../lib/api';

const { TextArea } = Input;
const { Option } = Select;

const MainPage: React.FC = () => {
    const [localRnaSequence, setLocalRnaSequence] = useState("TCAGGAGTTCGAGACCAGCCTGATCAACATGACGAAACCCTATCTCTACTAAAAATACAAAAATTAGCCGGGCGTGGTGGCATGCGCCTGTAGTCTCAGCTACTTGGGAGGCTGAAGCAGGAGAATCGTTTGAACCCAGGAGGCAGAGGTTGCAGTGAGCCGAGATCGTGCCACTGCACTCCAGCCTGGGTGACACAGCGAGACTCTGTCTCAAAAAAATAAAAATAAAAAAATAAATAAATAACCTTTAATTTAGTGAGACTTCATATAGAATTGTTTTAATGTTTAATATAGACCATTTGTTTTAGGTGAATTTAACAATTTCATACTGTGATTAAGATTAATTTCTTTTTCTGACTTCTACCAGAAAGCAGGAATTATGTTTCAAATGGACAATCATTTACCAAACCTTGTTAATCTGAATGAAGATCCACAACTATCTGAGATGCTGCTATATATGATAAAAGAAGGAACAACTACAGTTGGAAAGTATAAACCAAACTCAAGCCATGATATTCAGTTATCTGGGGTGCTGATTGCTGATGATCATTGGTATGTTAATCCTCTAAAAAAAAAGAAAAGGCACCTGTTCTATATCTTGATAACATGTGGTTTCCTTCATATGGCATATTCGTTGATACTGATCGTTTGGTAGAATTCTTCAAACCCATTGTTTAGTCAGGAAAAACATACATTCTGAGTGTGTTATAAGGATGATAGGTCAGTTACTCTCAATATAAAGTACAGTGTAATGCTCTCTCTGTTTTTGTTTTGGCATACTTGATCTGTTGATTGAAGAATAATTTATTTTCTTGCAATTATAATGATGCACATGCAAGTAAACTATCTATCTTACATAACAGAATTTTTGGTTGGATTGACCAATTTAAAAATGTTACTTTATGTGAATTTTGTTCATATGAATGGAATACTTGTATATATTGTTGGAATGATAGCGTATGTAAACTTTTTTGACTCTGCATTGTGTTTCCAAGATTTGT");

    const handleSequenceChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        const value = e.target.value.toUpperCase();
        // Filter only ACGTUN characters
        const filtered = value.replace(/[^ACGTUN]/g, '');
        // Limit to 1001 characters
        const truncated = filtered.slice(0, 1001);
        setLocalRnaSequence(truncated);
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleSubmit();
        }
    };
    const [localServer, setLocalServer] = useState('server1');
    const { setRnaSequence, setServer } = useRna();
    const navigate = useNavigate();
    const { t, changeLanguage, language } = useTranslation();

    const handleSubmit = async () => {
        try {
            setRnaSequence(localRnaSequence);
            setServer(localServer);

            // Submit task to backend for async processing
            const data = await submitTask({
                userId: 'user1',
                rnaSequence: localRnaSequence
            });

            // Navigate to results page with jobId for polling
            navigate(`/results/${data.jobId}`);
        } catch (error: any) {
            console.error('Error submitting task:', error);
            message.error(t('Submit task failed: {message}').replace('{message}', error.message));
        }
    };

    return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '80vh' }}>
            <Card
                title={
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span>{t('RNA Sequence Analysis')}</span>
                        <div>
                            <Button
                                onClick={() => changeLanguage('zh')}
                                disabled={language === 'zh'}
                                size="small"
                                type={language === 'zh' ? 'primary' : 'default'}
                            >
                                中文
                            </Button>
                            <Button
                                onClick={() => changeLanguage('en')}
                                disabled={language === 'en'}
                                size="small"
                                type={language === 'en' ? 'primary' : 'default'}
                                style={{ marginLeft: 8 }}
                            >
                                EN
                            </Button>
                        </div>
                    </div>
                }
                style={{ width: 800 }}
            >
                <div style={{ marginBottom: 8, textAlign: 'right', fontSize: 14, color: '#666' }}>
                    Only "A/C/G/T/U/N" characters are support. {localRnaSequence.length}/1001
                </div>
                <TextArea
                    rows={6}
                    value={localRnaSequence}
                    onChange={handleSequenceChange}
                    onKeyDown={handleKeyDown}
                    placeholder={t('Enter RNA sequence (ACGTUN only)')}
                    maxLength={1001}
                />
                <div style={{ margin: '20px 0' }}>
                    <Select value={localServer} onChange={(value) => setLocalServer(value)} style={{ width: '100%' }}>
                        <Option value="server1">{t('Server 1')}</Option>
                        <Option value="server2">{t('Server 2')}</Option>
                        <Option value="server3">{t('Server 3')}</Option>
                    </Select>
                </div>
                <Space direction="vertical" style={{ width: '100%' }} size="middle">
                    <Button type="primary" onClick={handleSubmit} block>
                        {t('Submit')}
                    </Button>
                </Space>
            </Card>
        </div>
    );
};

export default MainPage;
