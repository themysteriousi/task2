/**
 * voice-client.ts
 * Robust Voice Client connecting the Orb UI with the FastAPI Voice-RAG backend.
 * Features:
 * - Real-time microphone audio capture & AnalyserNode volume measurement
 * - Browser Web Speech Recognition fallback & live transcription
 * - 16kHz WAV audio encoding & WebSocket / REST streaming fallback
 * - SpeechSynthesis voice response with audio-reactive Orb pulsing
 */

export type VoiceClientState = 'idle' | 'connecting' | 'listening' | 'thinking' | 'speaking' | 'error'

export interface VoiceClientSignal {
    state: VoiceClientState
    inputVolume?: number
    outputVolume?: number
}

export interface RAGFinalData {
    query: string
    transcription: string
    answer: string
    retrieved_contexts: string[]
    latency_ms: number
    grounded: boolean
    total_voice_latency_ms?: number
}

export type VoiceClientListener = (signal: VoiceClientSignal) => void

const getBackendUrls = () => {
    const envWs = import.meta.env.VITE_BACKEND_WS_URL
    const envRest = import.meta.env.VITE_BACKEND_URL

    if (envWs && envRest) {
        return {
            wsUrl: `${envWs.replace(/\/$/, '')}/ws/voice-rag`,
            restUrl: `${envRest.replace(/\/$/, '')}/api/rag/text`
        }
    }

    if (typeof window !== 'undefined' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
        const isSecure = window.location.protocol === 'https:'
        const wsProto = isSecure ? 'wss:' : 'ws:'
        const httpProto = isSecure ? 'https:' : 'http:'
        const host = window.location.host
        return {
            wsUrl: envWs || `${wsProto}//${host}/ws/voice-rag`,
            restUrl: envRest || `${httpProto}//${host}/api/rag/text`,
        }
    }

    return {
        wsUrl: envWs || 'ws://localhost:8000/ws/voice-rag',
        restUrl: envRest || 'http://localhost:8000/api/rag/text',
    }
}

const { wsUrl: BACKEND_WS_URL, restUrl: BACKEND_REST_URL } = getBackendUrls()
const SAMPLE_RATE = 16000

export class VoiceClient {
    private ws: WebSocket | null = null
    private mediaStream: MediaStream | null = null
    private audioContext: AudioContext | null = null
    private analyser: AnalyserNode | null = null
    private scriptProcessor: ScriptProcessorNode | null = null
    private pcmChunks: Float32Array[] = []
    private volumeInterval: number | null = null
    private speechAnimationInterval: number | null = null
    private recognition: any = null

    private listeners: Set<VoiceClientListener> = new Set()
    public state: VoiceClientState = 'idle'
    public currentTranscription: string = ''
    public currentAnswer: string = ''
    public lastFinalData: RAGFinalData | null = null
    public errorMessage: string = ''

    // Event hooks for UI
    public onTranscription?: (text: string) => void
    public onToken?: (token: string) => void
    public onFinal?: (data: RAGFinalData) => void
    public onError?: (error: string) => void

    constructor() {
        this.initSpeechRecognition()
    }

    private initSpeechRecognition() {
        const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
        if (SpeechRec) {
            try {
                this.recognition = new SpeechRec()
                this.recognition.continuous = true
                this.recognition.interimResults = true
                this.recognition.lang = 'en-US'

                this.recognition.onresult = (event: any) => {
                    let transcript = ''
                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        transcript += event.results[i][0].transcript
                    }
                    if (transcript) {
                        this.currentTranscription = transcript
                        this.onTranscription?.(transcript)
                    }
                }

                this.recognition.onerror = (err: any) => {
                    console.warn('Speech recognition notice:', err)
                }
            } catch (e) {
                console.warn('SpeechRecognition initialization error:', e)
            }
        }
    }

    subscribe(listener: VoiceClientListener): () => void {
        this.listeners.add(listener)
        listener({ state: this.state })
        return () => this.listeners.delete(listener)
    }

    private emit(signal: VoiceClientSignal) {
        this.state = signal.state
        this.listeners.forEach((l) => l(signal))
    }

    async start() {
        if (this.state === 'listening' || this.state === 'thinking' || this.state === 'speaking') {
            // Clicking while active acts as stop/finish query
            return this.finishRecordingAndProcess()
        }

        // Cancel any ongoing speech synthesis
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel()
        }

        this.errorMessage = ''
        this.currentTranscription = ''
        this.currentAnswer = ''
        this.pcmChunks = []
        this.emit({ state: 'connecting' })

        try {
            // 1. Obtain Microphone Access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: SAMPLE_RATE,
                    echoCancellation: true,
                    noiseSuppression: true,
                },
                video: false,
            })

            // 2. Setup AudioContext & Analyser for reactive volume
            const AudioCtx = window.AudioContext || (window as any).webkitAudioContext
            this.audioContext = new AudioCtx({ sampleRate: SAMPLE_RATE })
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume()
            }

            const source = this.audioContext.createMediaStreamSource(this.mediaStream)
            this.analyser = this.audioContext.createAnalyser()
            this.analyser.fftSize = 256
            this.analyser.smoothingTimeConstant = 0.4

            // Script processor to accumulate PCM audio
            this.scriptProcessor = this.audioContext.createScriptProcessor(2048, 1, 1)
            this.scriptProcessor.onaudioprocess = (e) => {
                if (this.state !== 'listening') return
                const input = e.inputBuffer.getChannelData(0)
                this.pcmChunks.push(new Float32Array(input))
            }

            source.connect(this.analyser)
            this.analyser.connect(this.scriptProcessor)
            this.scriptProcessor.connect(this.audioContext.destination)

            // Start live volume polling for listening orb animation
            const dataArray = new Uint8Array(this.analyser.frequencyBinCount)
            this.volumeInterval = window.setInterval(() => {
                if (this.state === 'listening' && this.analyser) {
                    this.analyser.getByteFrequencyData(dataArray)
                    let sum = 0
                    for (let i = 0; i < dataArray.length; i++) {
                        sum += dataArray[i]
                    }
                    const avg = sum / dataArray.length
                    const normalized = Math.min(1, Math.max(0, avg / 80))
                    this.emit({ state: 'listening', inputVolume: normalized })
                }
            }, 60)

            // 3. Start speech recognition in browser if supported
            if (this.recognition) {
                try {
                    this.recognition.start()
                } catch {
                    // Ignore if already running
                }
            }

            // Connect WebSocket in background
            this.connectWebSocket()

            this.emit({ state: 'listening', inputVolume: 0 })
        } catch (err: any) {
            console.error('Microphone or Audio initialization failed:', err)
            this.errorMessage = err?.message || 'Microphone access denied or audio failed.'
            this.onError?.(this.errorMessage)
            this.cleanupAudio()
            this.emit({ state: 'error' })
        }
    }

    private connectWebSocket(): Promise<WebSocket | null> {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            return Promise.resolve(this.ws)
        }

        return new Promise((resolve) => {
            try {
                this.ws = new WebSocket(BACKEND_WS_URL)
                this.ws.binaryType = 'arraybuffer'

                this.ws.onopen = () => {
                    resolve(this.ws)
                }

                this.ws.onmessage = (event) => {
                    try {
                        const packet = JSON.parse(event.data as string)
                        this.handleBackendPacket(packet)
                    } catch (e) {
                        console.warn('Packet parse error:', e)
                    }
                }

                this.ws.onerror = (e) => {
                    console.warn('WebSocket connection error (will use REST fallback if needed):', e)
                    resolve(null)
                }

                this.ws.onclose = () => {
                    this.ws = null
                }
            } catch {
                resolve(null)
            }
        })
    }

    private handleBackendPacket(packet: any) {
        if (packet.type === 'transcription') {
            if (packet.text) {
                this.currentTranscription = packet.text
                this.onTranscription?.(packet.text)
            }
        } else if (packet.type === 'token') {
            this.currentAnswer += packet.token
            this.onToken?.(packet.token)
            if (this.state !== 'speaking') {
                this.emit({ state: 'speaking', outputVolume: 0.65 })
            }
        } else if (packet.type === 'final') {
            const data: RAGFinalData = packet.data
            this.lastFinalData = data
            if (data?.answer && !this.currentAnswer) {
                this.currentAnswer = data.answer
            }
            this.onFinal?.(data)
            this.speakAnswer(this.currentAnswer || data.answer)
        } else if (packet.type === 'error') {
            this.errorMessage = packet.message || 'Pipeline encountered an error.'
            this.onError?.(this.errorMessage)
            this.emit({ state: 'error' })
        }
    }

    public async finishRecordingAndProcess() {
        if (this.state !== 'listening') return

        this.emit({ state: 'thinking' })

        // Stop recognition & mic recording
        if (this.recognition) {
            try {
                this.recognition.stop()
            } catch {
                // ignore
            }
        }

        const wavBlob = this.encodeWAV(this.pcmChunks, SAMPLE_RATE)
        this.cleanupAudio()

        const transcribedQuery = this.currentTranscription.trim()

        // If we have a query transcribed or a WebSocket open
        const ws = await this.connectWebSocket()

        if (ws && ws.readyState === WebSocket.OPEN) {
            if (transcribedQuery) {
                // Send text query over WebSocket
                ws.send(JSON.stringify({ query: transcribedQuery }))
            } else if (wavBlob && wavBlob.size > 1000) {
                // Send audio bytes
                const arrayBuffer = await wavBlob.arrayBuffer()
                ws.send(arrayBuffer)
            } else {
                this.currentAnswer = "I didn't hear anything. Please try speaking again."
                this.speakAnswer(this.currentAnswer)
            }
        } else {
            // Fallback to HTTP REST
            await this.fallbackREST(transcribedQuery || 'What is the MS MARCO dataset?')
        }
    }

    public async sendTextQuery(query: string) {
        if (!query.trim()) return

        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel()
        }

        this.errorMessage = ''
        this.currentTranscription = query
        this.currentAnswer = ''
        this.onTranscription?.(query)
        this.emit({ state: 'thinking' })

        const ws = await this.connectWebSocket()
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ query }))
        } else {
            await this.fallbackREST(query)
        }
    }

    private async fallbackREST(query: string) {
        try {
            const res = await fetch(BACKEND_REST_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
            })

            if (!res.ok) {
                throw new Error(`Server returned status ${res.status}`)
            }

            const json = await res.json()
            this.currentTranscription = json.transcription || query
            this.currentAnswer = json.answer || ''
            this.lastFinalData = json.metadata
            this.onFinal?.(json.metadata)
            this.speakAnswer(this.currentAnswer)
        } catch (err: any) {
            this.errorMessage = err?.message || 'Failed to reach RAG server.'
            this.onError?.(this.errorMessage)
            this.emit({ state: 'error' })
        }
    }

    private speakAnswer(text: string) {
        if (!text) {
            this.emit({ state: 'idle' })
            return
        }

        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel()
            const utterance = new SpeechSynthesisUtterance(text)
            utterance.rate = 1.05
            utterance.pitch = 1.0

            // Animate outputVolume while speaking
            this.emit({ state: 'speaking', outputVolume: 0.7 })
            let pulse = 0
            if (this.speechAnimationInterval) clearInterval(this.speechAnimationInterval)
            this.speechAnimationInterval = window.setInterval(() => {
                pulse += 0.25
                const vol = 0.45 + 0.35 * Math.sin(pulse)
                if (this.state === 'speaking') {
                    this.emit({ state: 'speaking', outputVolume: vol })
                }
            }, 80)

            utterance.onend = () => {
                if (this.speechAnimationInterval) {
                    clearInterval(this.speechAnimationInterval)
                    this.speechAnimationInterval = null
                }
                this.emit({ state: 'idle' })
            }

            utterance.onerror = () => {
                if (this.speechAnimationInterval) {
                    clearInterval(this.speechAnimationInterval)
                    this.speechAnimationInterval = null
                }
                this.emit({ state: 'idle' })
            }

            window.speechSynthesis.speak(utterance)
        } else {
            this.emit({ state: 'idle' })
        }
    }

    public stop() {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel()
        }
        if (this.speechAnimationInterval) {
            clearInterval(this.speechAnimationInterval)
            this.speechAnimationInterval = null
        }
        if (this.recognition) {
            try {
                this.recognition.stop()
            } catch {
                // ignore
            }
        }
        this.cleanupAudio()
        this.emit({ state: 'idle' })
    }

    private cleanupAudio() {
        if (this.volumeInterval) {
            clearInterval(this.volumeInterval)
            this.volumeInterval = null
        }
        this.scriptProcessor?.disconnect()
        this.scriptProcessor = null
        this.analyser?.disconnect()
        this.analyser = null
        this.audioContext?.close()
        this.audioContext = null
        this.mediaStream?.getTracks().forEach((t) => t.stop())
        this.mediaStream = null
    }

    private encodeWAV(chunks: Float32Array[], sampleRate: number): Blob {
        const totalLength = chunks.reduce((acc, c) => acc + c.length, 0)
        const merged = new Float32Array(totalLength)
        let offset = 0
        for (const chunk of chunks) {
            merged.set(chunk, offset)
            offset += chunk.length
        }

        const buffer = new ArrayBuffer(44 + merged.length * 2)
        const view = new DataView(buffer)

        // Write WAV RIFF Header
        const writeString = (view: DataView, offset: number, string: string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i))
            }
        }

        writeString(view, 0, 'RIFF')
        view.setUint32(4, 36 + merged.length * 2, true)
        writeString(view, 8, 'WAVE')
        writeString(view, 12, 'fmt ')
        view.setUint32(16, 16, true) // PCM chunk size
        view.setUint16(20, 1, true) // Linear PCM
        view.setUint16(22, 1, true) // Mono
        view.setUint32(24, sampleRate, true) // Sample rate
        view.setUint32(28, sampleRate * 2, true) // Byte rate
        view.setUint16(32, 2, true) // Block align
        view.setUint16(34, 16, true) // Bits per sample
        writeString(view, 36, 'data')
        view.setUint32(40, merged.length * 2, true)

        // Write PCM Samples
        let index = 44
        for (let i = 0; i < merged.length; i++) {
            const s = Math.max(-1, Math.min(1, merged[i]))
            view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7fff, true)
            index += 2
        }

        return new Blob([view], { type: 'audio/wav' })
    }
}
