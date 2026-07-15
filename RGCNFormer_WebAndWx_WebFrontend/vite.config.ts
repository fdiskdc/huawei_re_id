/*
 * @Author: Chao Deng && chaodeng987@outlook.com
 * @Date: 2026-01-20 12:06:34
 * @LastEditors: Chao Deng && chaodeng987@outlook.com
 * @LastEditTime: 2026-01-21 09:44:03
 * @FilePath: /rgcnformer_mobile_web/frontend/vite.config.ts
 * @Description: 
 * 那只是一场游戏一场梦
 *  
 * https://orcid.org/0009-0009-8520-1656
 * DOI: 10.3390/app15158626
 * DOI: 10.3390/rs17142354
 * Copyright (c) 2026 by ${Chao Deng}, All Rights Reserved. 
 */
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: '/rgcnformer/',
  plugins: [react()],
  server: {
    proxy: {
      // 将所有 /rgcnformer/api 开头的请求代理到后端
      '/rgcnformer/api': {
        target: process.env.VITE_BACKEND_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/rgcnformer/, '/mrmodn'),
      },
    },
  },
})
