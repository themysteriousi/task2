import type { OrbAdapter, OrbSignal, OrbSignalListener, OrbState } from '../types'

export interface CustomBackendAdapterConfig {
  /** WebSocket URL for the voice RAG endpoint. Defaults to '/ws/voice-rag'. */
  wsUrl?: string
  /** Called when the WebSocket connection is established. */
  onConnect?: () => void
  /** Called when the WebSocket connection closes. */
  onDisconnect?: () => void
  /** Called when an error occurs. */
  onError?: (error: Error) => void
}

interface BackendMessage {
  type: 'transcription' | 'token' | 'final' | 'error'
  text?: string
  token?: string
  data?: unknown
  message?: string
}

function mapState(backendState: 'connecting' | 'listening' | 'thinking' | 'speaking' | 'idle' | 'error'): OrbState {
  switch (backendState) {
    case 'connecting':
      return 'connecting'
    case 'listening':
      return 'listening'
    case 'thinking':
      return 'thinking'
    case 'speaking':
      return 'speaking'
    case 'error':
      return 'error'
    default:
      return 'idle'
  }
}

export function createCustomBackendAdapter(config: CustomBackendAdapterConfig = {}): OrbAdapter {
  const { wsUrl = '/ws/voice-rag', onConnect, onDisconnect, onError } = config
  let ws: WebSocket | null = null
  let mediaRecorder: MediaRecorder | null = null
  let audioChunks: Blob[] = []
  let isRecording = false
  let currentSignal: OrbSignal = { state: 'idle', volume: 0 }
  let listeners = new Set<OrbSignalListener>()
  let thinkingTimeout: ReturnType<typeof setTimeout> | null = null

  function emit(signal: OrbSignal) {
    currentSignal = signal
    listeners.forEach((listener) => listener(signal))
  }

  function emitPatch(patch: Partial<OrbSignal>) {
    emit({ ...currentSignal, ...patch })
  }

  function connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const fullUrl = wsUrl.startsWith('ws') ? wsUrl : `${protocol}//${window.location.host}${wsUrl}`

      ws = new WebSocket(fullUrl)

      ws.onopen = () => {
        onConnect?.()
        emitPatch({ state: 'connecting' })
        resolve()
      }

      ws.onclose = () => {
        onDisconnect?.()
        stopRecording()
        emitPatch({ state: 'idle', volume: 0, inputVolume: 0, outputVolume: 0 })
      }

      ws.onerror = (event) => {
        const error = new Error('WebSocket connection failed')
        onError?.(error)
        emitPatch({ state: 'error', error, volume: 0 })
        reject(error)
      }

      ws.onmessage = (event) => {
        try {
          const message: BackendMessage = JSON.parse(event.data)
          handleBackendMessage(message)
        } catch (err) {
          console.error('[custom-backend] Failed to parse message:', err)
        }
      }
    })
  }

  function handleBackendMessage(message: BackendMessage) {
    switch (message.type) {
      case 'transcription': {
        // User finished speaking, we got transcription
        emitPatch({ state: 'thinking', inputVolume: 0 })
        if (thinkingTimeout) clearTimeout(thinkingTimeout)
        thinkingTimeout = setTimeout(() => {
          emitPatch({ state: 'speaking', outputVolume: 0.3 })
        }, 300)
        break
      }
      case 'token': {
        // Assistant is speaking - streaming tokens
        emitPatch({ state: 'speaking', outputVolume: 0.5 })
        break
      }
      case 'final': {
        // Response complete
        emitPatch({ state: 'speaking', outputVolume: 0.8 })
        setTimeout(() => {
          emitPatch({ state: 'listening', outputVolume: 0, volume: 0 })
        }, 1000)
        break
      }
      case 'error': {
        const error = new Error(message.message || 'Backend error')
        onError?.(error)
        emitPatch({ state: 'error', error, volume: 0 })
        break
      }
    }
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' })

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        if (audioChunks.length === 0 || !ws || ws.readyState !== WebSocket.OPEN) {
          audioChunks = []
          return
        }

        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' })
        audioChunks = []
        const arrayBuffer = await audioBlob.arrayBuffer()
        ws.send(arrayBuffer)
      }

      isRecording = true
      mediaRecorder.start(500) // Send chunks every 500ms
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to start recording')
      onError?.(error)
      emitPatch({ state: 'error', error })
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop()
      mediaRecorder.stream.getTracks().forEach((track) => track.stop())
    }
    isRecording = false
  }

  return {
    async start() {
      await connectWebSocket()
      await startRecording()
    },

    stop() {
      if (thinkingTimeout) {
        clearTimeout(thinkingTimeout)
        thinkingTimeout = null
      }
      stopRecording()
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close()
      }
    },

    subscribe(listener: OrbSignalListener) {
      listeners.add(listener)
      listener(currentSignal)

      return () => {
        listeners.delete(listener)
      }
    },
  }
}