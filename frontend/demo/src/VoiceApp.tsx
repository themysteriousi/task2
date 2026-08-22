/**
 * VoiceApp.tsx
 * Custom voice interface that wires the Orb UI with the FastAPI RAG backend
 * using a custom OrbAdapter powered by VoiceClient.
 */
import { useMemo, useRef } from 'react'
import { Orb } from 'orb-ui'
import type { OrbAdapter, OrbSignal } from 'orb-ui'
import { VoiceClient } from './voice-client'

function createBackendAdapter(client: VoiceClient): OrbAdapter {
    return {
        subscribe(listener: (signal: OrbSignal) => void) {
            return client.subscribe((signal) => {
                const state = signal.state
                if (state === 'listening') {
                    listener({ state, inputVolume: signal.inputVolume ?? 0 })
                } else if (state === 'speaking') {
                    listener({ state, outputVolume: signal.outputVolume ?? 0.7 })
                } else {
                    listener({ state })
                }
            })
        },
        start: () => client.start(),
        stop: () => client.stop(),
    }
}

export default function VoiceApp() {
    const clientRef = useRef<VoiceClient | null>(null)
    if (!clientRef.current) {
        clientRef.current = new VoiceClient()
    }

    const adapter = useMemo(() => createBackendAdapter(clientRef.current!), [])

    return (
        <div
            style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '100vh',
                background: '#0a0a0a',
                color: '#fff',
                fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                gap: '28px',
            }}
        >
            {/* Glow background */}
            <div
                style={{
                    position: 'fixed',
                    inset: 0,
                    background:
                        'radial-gradient(circle at 50% 40%, rgba(82,156,255,0.10), transparent 55%)',
                    pointerEvents: 'none',
                }}
            />

            {/* Header */}
            <div style={{ textAlign: 'center', zIndex: 1 }}>
                <p
                    style={{
                        margin: 0,
                        fontSize: 11,
                        letterSpacing: '0.14em',
                        textTransform: 'uppercase',
                        color: '#78b9f2',
                        fontWeight: 650,
                    }}
                >
                    Voice RAG Pipeline
                </p>
                <h1
                    style={{
                        margin: '10px 0 0',
                        fontSize: 'clamp(28px, 5vw, 48px)',
                        fontWeight: 800,
                        letterSpacing: '-0.04em',
                        background: 'linear-gradient(100deg, #fff 10%, #a9d6ff 90%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                    }}
                >
                    Ask Anything
                </h1>
                <p style={{ margin: '10px 0 0', color: '#666', fontSize: 14, lineHeight: 1.6 }}>
                    Speak your question — powered by Nemotron + MSMARCO RAG
                </p>
            </div>

            {/* Orb */}
            <div style={{ zIndex: 1 }}>
                <Orb
                    adapter={adapter}
                    theme="cloud"
                    aria-label="Start voice assistant"
                    style={{ width: 180, height: 180 }}
                />
            </div>

            {/* Hint */}
            <p
                style={{
                    margin: 0,
                    fontSize: 12,
                    color: '#444',
                    letterSpacing: '0.04em',
                    zIndex: 1,
                }}
            >
                Click the orb to start · click again to stop
            </p>

            {/* Backend info badge */}
            <div
                style={{
                    position: 'fixed',
                    bottom: 24,
                    right: 24,
                    background: 'rgba(16,16,16,0.9)',
                    border: '1px solid #222',
                    borderRadius: 10,
                    padding: '8px 14px',
                    fontSize: 11,
                    color: '#555',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    zIndex: 10,
                }}
            >
                <span
                    style={{
                        width: 6,
                        height: 6,
                        borderRadius: '50%',
                        background: '#3fba74',
                        boxShadow: '0 0 8px rgba(63,186,116,0.8)',
                        flexShrink: 0,
                    }}
                />
                ws://localhost:8000/ws/voice-rag
            </div>
        </div>
    )
}
