import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { RnaProvider } from './context/RnaContext'
import { LanguageProvider } from './lib/i18n/LanguageContext'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter basename="/rgcnformer">
      <RnaProvider>
        <LanguageProvider>
          <App />
        </LanguageProvider>
      </RnaProvider>
    </BrowserRouter>
  </StrictMode>,
)
