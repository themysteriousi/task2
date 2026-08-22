import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import VoiceApp from './VoiceApp'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <VoiceApp />
  </StrictMode>,
)
