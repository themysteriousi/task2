/**
 * voice-client.ts
 * Custom voice client that connects the orb-ui frontend to the FastAPI RAG backend.
 * Protocol:
 *   SEND:   raw PCM audio bytes (ArrayBuffer)
 *   RECEIVE: { type: "transcription", text } | { type: "token", token } | { type: "final", data } | { type: "error", message }
 */

export type VoiceClientState = 'idle' | 'connecting' | 'listening' | 'thinking' | 'speaking' | 'error'

export interface VoiceClientSignal {
    state: VoiceClientState
    inputVolume?: number
    outputVolume?: number
}

export type VoiceClientListener = (signal: VoiceClientSignal) => void

const BACKEND_WS_URL = 'ws://localhost:8000/ws/voice-rag'
const AUDIO_SAMPLE_RATE = 16000
const CHUNK_INTERVAL_MS = 250

export class VoiceClient {
    private ws: WebSocket | null = null
    private mediaStream: MediaStream | null = null
    private audioContext: AudioContext | null = null
    private processor: ScriptProcessorNode | null = null
    private listeners: Set<VoiceClientListener> = new Set()
    private state: VoiceClientState = 'idle'

    subscribe(listener: VoiceClientListener): () => void {
        this.listeners.add(listener)
        // Immediately emit current state
        listener({ state: this.state })
        return () => this.listeners.delete(listener)
    }

    private emit(signal: VoiceClientSignal) {
        this.state = signal.state
        this.listeners.forEach((l) => l(signal))
    }

    async start() {
        if (this.state !== 'idle' && this.state !== 'error') return

        this.emit({ state: 'connecting' })

        try {
            // 1. Request microphone
            this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })

            // 2. Connect WebSocket
            this.ws = new WebSocket(BACKEND_WS_URL)
            this.ws.binaryType = 'arraybuffer'

            await new Promise<void>((resolve, reject) => {
                this.ws!.onopen = () => resolve()
                this.ws!.onerror = () => reject(new Error('WebSocket connection failed'))
            })

            // 3. Parse incoming messages
            this.ws.onmessage = (event) => {
                try {
                    const packet = JSON.parse(event.data as string)
                    if (packet.type === 'transcription') {
                        // User finished speaking, now thinking
                        this.emit({ state: 'thinking' })
                    } else if (packet.type === 'token') {
                        // Backend streaming answer tokens
                        this.emit({ state: 'speaking', outputVolume: 0.7 })
                    } else if (packet.type === 'final') {
                        // Answer complete, return to listening
                        this.emit({ state: 'listening', inputVolume: 0 })
                    } else if (packet.type === 'error') {
                        console.error('Backend error:', packet.message)
                        this.emit({ state: 'error' })
                    }
                } catch {
                    // ignore parse errors
                }
            }

            this.ws.onclose = () => {
                this.cleanup()
                this.emit({ state: 'idle' })
            }

            this.ws.onerror = () => {
                this.emit({ state: 'error' })
                this.cleanup()
            }

            // 4. Stream microphone audio
            this.audioContext = new AudioContext({ sampleRate: AUDIO_SAMPLE_RATE })
            const source = this.audioContext.createMediaStreamSource(this.mediaStream)
            this.processor = this.audioContext.createScriptProcessor(4096, 1, 1)

            let buffer: Float32Array[] = []
            let lastSend = Date.now()

            this.processor.onaudioprocess = (e) => {
                if (this.ws?.readyState !== WebSocket.OPEN) return

                const pcm = e.inputBuffer.getChannelData(0)
                buffer.push(new Float32Array(pcm))

                if (Date.now() - lastSend >= CHUNK_INTERVAL_MS) {
                    // Flatten and convert to Int16 PCM
                    const totalLength = buffer.reduce((n, a) => n + a.length, 0)
                    const flat = new Float32Array(totalLength)
                    let offset = 0
                    for (const chunk of buffer) {
                        flat.set(chunk, offset)
                        offset += chunk.length
                    }

                    const int16 = new Int16Array(flat.length)
                    for (let i = 0; i < flat.length; i++) {
                        int16[i] = Math.max(-32768, Math.min(32767, flat[i] * 32768))
                    }

                    this.ws.send(int16.buffer)
                    buffer = []
                    lastSend = Date.now()

                    // Calculate volume for visual feedback
                    const rms = Math.sqrt(flat.reduce((sum, s) => sum + s * s, 0) / flat.length)
                    this.emit({ state: 'listening', inputVolume: Math.min(rms * 8, 1) })
                }
            }

            source.connect(this.processor)
            this.processor.connect(this.audioContext.destination)

            this.emit({ state: 'listening', inputVolume: 0 })
        } catch (err) {
            console.error('VoiceClient start error:', err)
            this.emit({ state: 'error' })
            this.cleanup()
        }
    }

    stop() {
        this.cleanup()
        this.emit({ state: 'idle' })
    }

    private cleanup() {
        this.processor?.disconnect()
        this.processor = null
        this.audioContext?.close()
        this.audioContext = null
        this.mediaStream?.getTracks().forEach((t) => t.stop())
        this.mediaStream = null
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.close()
        }
        this.ws = null
    }
}
