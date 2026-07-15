/**
 * eslint.config.js - ESLint 9 flat 配置 / ESLint 9 flat config
 *
 * ESLint 9 flat config(取代旧式 .eslintrc):对所有 TS/TSX 源文件应用 JS 推荐、
 * TS 推荐、React Hooks 与 React Refresh 规则;全局忽略 dist。 / ESLint 9 flat
 * config (replaces legacy .eslintrc). Applies JS recommended, TS recommended,
 * React Hooks, and React Refresh rules to all TS/TSX files; globally ignores dist.
 *
 * 功能模块 / Modules:
 * - globalIgnores: 忽略 dist 构建产物 / Ignore dist build output
 * - js.configs.recommended: JS 基础规则 / JS base rules
 * - tseslint.configs.recommended: TS 规则 / TS rules
 * - reactHooks.configs.flat.recommended: React Hooks 规则 / React Hooks rules
 * - reactRefresh.vite: Vite HMR 兼容规则 / Vite HMR rules
 *
 * 输入 / Inputs:
 * - 无显式入参,ESLint 启动时自动加载 / None, auto-loaded by ESLint
 *
 * 输出 / Outputs:
 * - ESLint 配置对象 / ESLint config object
 *
 * 数据流 / Data Flow:
 * 1. ESLint 启动 → 读本文件 / ESLint reads this file on start
 * 2. 对匹配 TS/TSX 的文件应用规则 / Apply rules to matching TS/TSX files
 * 3. dist/ 永远不被 lint / dist/ is never linted
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: @eslint/js、globals、eslint-plugin-react-hooks、react-refresh、typescript-eslint
 * - 被调用 / Called by: `npm run lint`
 *
 * 使用示例 / Usage Example:
 *     npm run lint
 *
 * 作者 / Author: Chao Deng (chaodeng987@outlook.com)
 * 版本 / Version: 1.0
 */
import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
  },
])
