/**
 * VoiceApp.tsx
 * Premium Voice Agent Interface connecting orb-ui with FastAPI Voice-RAG pipeline.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { Orb } from 'orb-ui'
import type { OrbAdapter, OrbSignal, OrbTheme } from 'orb-ui'
import { Agentation } from 'agentation'
import { VoiceClient } from './voice-client'
import type { RAGFinalData, VoiceClientState } from './voice-client'

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
    const client = clientRef.current

    const [state, setState] = useState<VoiceClientState>('idle')
    const [volume, setVolume] = useState<number>(0)
    const [theme, setTheme] = useState<OrbTheme>('radial')
    const [transcription, setTranscription] = useState<string>('')
    const [answer, setAnswer] = useState<string>('')
    const [finalData, setFinalData] = useState<RAGFinalData | null>(null)
    const [error, setError] = useState<string>('')
    const [textInput, setTextInput] = useState<string>('')
    const [showSources, setShowSources] = useState<boolean>(false)

    useEffect(() => {
        const unsubscribe = client.subscribe((signal) => {
            setState(signal.state)
            if (signal.inputVolume !== undefined) setVolume(signal.inputVolume)
            else if (signal.outputVolume !== undefined) setVolume(signal.outputVolume)
            else setVolume(0)
        })

        client.onTranscription = (text) => {
            setTranscription(text)
            setError('')
        }

        client.onToken = (_token) => {
            setAnswer(client.currentAnswer)
        }

        client.onFinal = (data) => {
            setFinalData(data)
            setAnswer(data.answer || client.currentAnswer)
        }

        client.onError = (err) => {
            setError(err)
        }

        return () => {
            unsubscribe()
            client.stop()
        }
    }, [client])

    const adapter = useMemo(() => createBackendAdapter(client), [client])

    const handleSendText = (e?: React.FormEvent) => {
        if (e) e.preventDefault()
        if (!textInput.trim()) return
        const q = textInput.trim()
        setTextInput('')
        setAnswer('')
        setFinalData(null)
        client.sendTextQuery(q)
    }

    const handlePromptClick = (prompt: string) => {
        setTextInput('')
        setAnswer('')
        setFinalData(null)
        client.sendTextQuery(prompt)
    }

    const getStatusBadge = () => {
        switch (state) {
            case 'listening':
                return {
                    label: 'Listening to your voice...',
                    color: '#3b82f6',
                    bg: 'rgba(59, 130, 246, 0.15)',
                    border: 'rgba(59, 130, 246, 0.3)',
                    dot: '#60a5fa',
                    animate: true,
                }
            case 'thinking':
                return {
                    label: 'Retrieving context & generating...',
                    color: '#a855f7',
                    bg: 'rgba(168, 85, 247, 0.15)',
                    border: 'rgba(168, 85, 247, 0.3)',
                    dot: '#c084fc',
                    animate: true,
                }
            case 'speaking':
                return {
                    label: 'Answering with voice...',
                    color: '#10b981',
                    bg: 'rgba(16, 185, 129, 0.15)',
                    border: 'rgba(16, 185, 129, 0.3)',
                    dot: '#34d399',
                    animate: true,
                }
            case 'connecting':
                return {
                    label: 'Connecting audio stream...',
                    color: '#f59e0b',
                    bg: 'rgba(245, 158, 11, 0.15)',
                    border: 'rgba(245, 158, 11, 0.3)',
                    dot: '#fbbf24',
                    animate: true,
                }
            case 'error':
                return {
                    label: error || 'Connection issue',
                    color: '#ef4444',
                    bg: 'rgba(239, 68, 68, 0.15)',
                    border: 'rgba(239, 68, 68, 0.3)',
                    dot: '#f87171',
                    animate: false,
                }
            default:
                return {
                    label: 'Click the Orb or speak to start',
                    color: '#94a3b8',
                    bg: 'rgba(148, 163, 184, 0.08)',
                    border: 'rgba(148, 163, 184, 0.18)',
                    dot: '#64748b',
                    animate: false,
                }
        }
    }

    const badge = getStatusBadge()

    return (
        <div
            style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'flex-start',
                minHeight: '100vh',
                background: 'radial-gradient(ellipse at 50% 15%, #111827 0%, #030712 100%)',
                color: '#f8fafc',
                fontFamily: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                padding: '40px 20px 80px 20px',
                boxSizing: 'border-box',
                position: 'relative',
                overflowX: 'hidden',
            }}
        >
            {/* Ambient Background Glow */}
            <div
                style={{
                    position: 'fixed',
                    inset: 0,
                    background:
                        state === 'listening'
                            ? 'radial-gradient(circle at 50% 38%, rgba(59, 130, 246, 0.18), transparent 60%)'
                            : state === 'speaking'
                            ? 'radial-gradient(circle at 50% 38%, rgba(16, 185, 129, 0.18), transparent 60%)'
                            : state === 'thinking'
                            ? 'radial-gradient(circle at 50% 38%, rgba(168, 85, 247, 0.18), transparent 60%)'
                            : 'radial-gradient(circle at 50% 38%, rgba(56, 189, 248, 0.08), transparent 50%)',
                    transition: 'background 0.8s ease',
                    pointerEvents: 'none',
                }}
            />

            {/* Top Navigation / Header */}
            <header
                style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    textAlign: 'center',
                    zIndex: 1,
                    maxWidth: 720,
                    width: '100%',
                    marginBottom: 20,
                }}
            >
                <div
                    style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 6,
                        padding: '4px 12px',
                        borderRadius: 20,
                        background: 'rgba(56, 189, 248, 0.1)',
                        border: '1px solid rgba(56, 189, 248, 0.25)',
                        fontSize: 12,
                        fontWeight: 600,
                        color: '#38bdf8',
                        letterSpacing: '0.06em',
                        textTransform: 'uppercase',
                        marginBottom: 12,
                    }}
                >
                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#38bdf8' }} />
                    Voice RAG Pipeline · Nemotron + MS MARCO
                </div>

                <h1
                    style={{
                        margin: 0,
                        fontSize: 'clamp(28px, 4.5vw, 44px)',
                        fontWeight: 800,
                        letterSpacing: '-0.03em',
                        background: 'linear-gradient(135deg, #ffffff 30%, #94a3b8 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                    }}
                >
                    Ask Anything with Voice
                </h1>
                <p style={{ margin: '8px 0 0', color: '#94a3b8', fontSize: 15, maxWidth: 520, lineHeight: 1.5 }}>
                    Real-time speech transcription, semantic vector retrieval & streaming neural voice responses.
                </p>

                {/* Theme Selector Pill */}
                <div
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                        background: 'rgba(15, 23, 42, 0.7)',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        borderRadius: 24,
                        padding: '4px 6px',
                        marginTop: 18,
                        backdropFilter: 'blur(8px)',
                    }}
                >
                    <span style={{ fontSize: 11, color: '#64748b', padding: '0 8px', fontWeight: 600 }}>THEME:</span>
                    {(['radial', 'cloud', 'circle', 'bars'] as OrbTheme[]).map((t) => (
                        <button
                            key={t}
                            type="button"
                            onClick={() => setTheme(t)}
                            style={{
                                appearance: 'none',
                                border: 0,
                                borderRadius: 16,
                                padding: '4px 12px',
                                fontSize: 12,
                                fontWeight: theme === t ? 600 : 400,
                                background: theme === t ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                                color: theme === t ? '#38bdf8' : '#94a3b8',
                                cursor: 'pointer',
                                transition: 'all 0.2s ease',
                            }}
                        >
                            {t.charAt(0).toUpperCase() + t.slice(1)}
                        </button>
                    ))}
                </div>
            </header>

            {/* Interactive Orb Stage */}
            <main
                style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    zIndex: 2,
                    margin: '12px 0 20px 0',
                }}
            >
                <div
                    role="button"
                    tabIndex={0}
                    onClick={() => client.start()}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault()
                            client.start()
                        }
                    }}
                    title={state === 'listening' ? 'Click to finish & search' : 'Click to start speaking'}
                    style={{
                        position: 'relative',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        width: 220,
                        height: 220,
                        borderRadius: '50%',
                        background: 'radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 70%)',
                        boxShadow:
                            state === 'listening'
                                ? '0 0 50px rgba(59, 130, 246, 0.35)'
                                : state === 'speaking'
                                ? '0 0 50px rgba(16, 185, 129, 0.35)'
                                : state === 'thinking'
                                ? '0 0 50px rgba(168, 85, 247, 0.35)'
                                : '0 0 30px rgba(0,0,0,0.4)',
                        transition: 'all 0.3s ease',
                        cursor: 'pointer',
                        transform: state === 'listening' ? 'scale(1.04)' : 'scale(1)',
                    }}
                >
                    <Orb
                        adapter={adapter}
                        theme={theme}
                        aria-label="Interactive voice assistant orb - Click to speak"
                        style={{ width: 200, height: 200, pointerEvents: 'none' }}
                    />
                </div>

                {/* Interactive Status Indicator & Action Button */}
                <button
                    type="button"
                    onClick={() => client.start()}
                    style={{
                        appearance: 'none',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 8,
                        background: badge.bg,
                        border: `1px solid ${badge.border}`,
                        color: badge.color,
                        borderRadius: 20,
                        padding: '10px 20px',
                        fontSize: 14,
                        fontWeight: 600,
                        marginTop: 16,
                        backdropFilter: 'blur(10px)',
                        transition: 'all 0.2s ease',
                        cursor: 'pointer',
                        boxShadow: state === 'listening' ? '0 0 16px rgba(59, 130, 246, 0.3)' : 'none',
                    }}
                >
                    <span
                        style={{
                            width: 8,
                            height: 8,
                            borderRadius: '50%',
                            background: badge.dot,
                            boxShadow: `0 0 8px ${badge.dot}`,
                            animation: badge.animate ? 'pulse 1.5s infinite ease-in-out' : 'none',
                        }}
                    />
                    {state === 'idle'
                        ? '🎤 Click Orb to Speak'
                        : state === 'listening'
                        ? '⏹️ Stop & Search'
                        : badge.label}
                </button>
            </main>

            {/* Quick Prompts */}
            {state === 'idle' && !transcription && !answer && (
                <div
                    style={{
                        display: 'flex',
                        flexWrap: 'wrap',
                        justifyContent: 'center',
                        gap: 8,
                        maxWidth: 640,
                        zIndex: 2,
                        marginBottom: 20,
                    }}
                >
                    {[
                        'What is the MS MARCO dataset?',
                        'How does hierarchical semantic chunking work?',
                        'Explain vector embeddings and retrieval',
                    ].map((prompt, i) => (
                        <button
                            key={i}
                            type="button"
                            onClick={() => handlePromptClick(prompt)}
                            style={{
                                appearance: 'none',
                                background: 'rgba(255, 255, 255, 0.04)',
                                border: '1px solid rgba(255, 255, 255, 0.1)',
                                borderRadius: 18,
                                padding: '6px 14px',
                                color: '#cbd5e1',
                                fontSize: 12,
                                cursor: 'pointer',
                                transition: 'all 0.2s ease',
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.background = 'rgba(56, 189, 248, 0.12)'
                                e.currentTarget.style.borderColor = 'rgba(56, 189, 248, 0.35)'
                                e.currentTarget.style.color = '#fff'
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.04)'
                                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)'
                                e.currentTarget.style.color = '#cbd5e1'
                            }}
                        >
                            💬 {prompt}
                        </button>
                    ))}
                </div>
            )}

            {/* Interaction Card (Transcriptions & AI Answer) */}
            {(transcription || answer || state === 'thinking') && (
                <div
                    style={{
                        width: '100%',
                        maxWidth: 680,
                        background: 'rgba(15, 23, 42, 0.65)',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        borderRadius: 16,
                        padding: 20,
                        backdropFilter: 'blur(16px)',
                        zIndex: 2,
                        marginBottom: 24,
                        boxShadow: '0 20px 40px -15px rgba(0,0,0,0.5)',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 16,
                    }}
                >
                    {/* User Question */}
                    {transcription && (
                        <div>
                            <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', marginBottom: 4 }}>
                                You Asked:
                            </div>
                            <div style={{ fontSize: 16, fontWeight: 600, color: '#f1f5f9', lineHeight: 1.4 }}>
                                "{transcription}"
                            </div>
                        </div>
                    )}

                    {/* AI Answer */}
                    <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                            <div style={{ fontSize: 11, fontWeight: 700, color: '#10b981', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: 6 }}>
                                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#10b981' }} />
                                Answer:
                            </div>
                            {finalData?.latency_ms && (
                                <div style={{ fontSize: 11, color: '#64748b', background: 'rgba(255,255,255,0.05)', padding: '2px 8px', borderRadius: 10 }}>
                                    ⚡ {Math.round(finalData.latency_ms)}ms latency
                                </div>
                            )}
                        </div>

                        {answer ? (
                            <div style={{ fontSize: 15, lineHeight: 1.6, color: '#e2e8f0', whiteSpace: 'pre-wrap' }}>
                                {answer}
                            </div>
                        ) : state === 'thinking' ? (
                            <div style={{ fontSize: 14, color: '#94a3b8', fontStyle: 'italic' }}>
                                Thinking & retrieving context from vector store...
                            </div>
                        ) : null}
                    </div>

                    {/* Retrieved Sources Accordion */}
                    {finalData?.retrieved_contexts && finalData.retrieved_contexts.length > 0 && (
                        <div style={{ borderTop: '1px solid rgba(255, 255, 255, 0.08)', paddingTop: 12 }}>
                            <button
                                type="button"
                                onClick={() => setShowSources(!showSources)}
                                style={{
                                    appearance: 'none',
                                    background: 'transparent',
                                    border: 0,
                                    padding: 0,
                                    fontSize: 12,
                                    fontWeight: 600,
                                    color: '#38bdf8',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 6,
                                }}
                            >
                                {showSources ? '▼ Hide' : '▶ Show'} {finalData.retrieved_contexts.length} Retrieved Context Sources
                            </button>

                            {showSources && (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 10 }}>
                                    {finalData.retrieved_contexts.map((ctx, idx) => (
                                        <div
                                            key={idx}
                                            style={{
                                                background: 'rgba(0, 0, 0, 0.3)',
                                                border: '1px solid rgba(255, 255, 255, 0.06)',
                                                borderRadius: 8,
                                                padding: 10,
                                                fontSize: 12,
                                                color: '#94a3b8',
                                                lineHeight: 1.5,
                                            }}
                                        >
                                            <span style={{ color: '#38bdf8', fontWeight: 600, marginRight: 6 }}>
                                                [{idx + 1}]
                                            </span>
                                            {ctx}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}

            {/* Bottom Text Input Bar */}
            <footer
                style={{
                    position: 'fixed',
                    bottom: 24,
                    width: 'calc(100% - 40px)',
                    maxWidth: 640,
                    zIndex: 10,
                }}
            >
                <form
                    onSubmit={handleSendText}
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        background: 'rgba(15, 23, 42, 0.85)',
                        border: '1px solid rgba(255, 255, 255, 0.15)',
                        borderRadius: 30,
                        padding: '6px 8px 6px 16px',
                        backdropFilter: 'blur(16px)',
                        boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
                    }}
                >
                    <input
                        type="text"
                        value={textInput}
                        onChange={(e) => setTextInput(e.target.value)}
                        placeholder="Type a question or click the orb above to talk..."
                        style={{
                            flex: 1,
                            background: 'transparent',
                            border: 0,
                            outline: 'none',
                            color: '#fff',
                            fontSize: 14,
                            fontFamily: 'inherit',
                        }}
                    />

                    {/* Mic Toggle Button */}
                    <button
                        type="button"
                        onClick={() => {
                            if (state === 'listening') client.finishRecordingAndProcess()
                            else client.start()
                        }}
                        title={state === 'listening' ? 'Stop Listening' : 'Start Microphone'}
                        style={{
                            appearance: 'none',
                            border: 0,
                            borderRadius: '50%',
                            width: 36,
                            height: 36,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: state === 'listening' ? '#ef4444' : 'rgba(255, 255, 255, 0.08)',
                            color: '#fff',
                            cursor: 'pointer',
                            transition: 'all 0.2s ease',
                        }}
                    >
                        {state === 'listening' ? '⏹' : '🎤'}
                    </button>

                    {/* Send Button */}
                    <button
                        type="submit"
                        disabled={!textInput.trim()}
                        style={{
                            appearance: 'none',
                            border: 0,
                            borderRadius: 20,
                            padding: '8px 16px',
                            background: textInput.trim() ? '#38bdf8' : 'rgba(255, 255, 255, 0.1)',
                            color: textInput.trim() ? '#0f172a' : '#64748b',
                            fontWeight: 600,
                            fontSize: 13,
                            cursor: textInput.trim() ? 'pointer' : 'default',
                            transition: 'all 0.2s ease',
                        }}
                    >
                        Send
                    </button>
                </form>
            </footer>

            {/* Agentation UI Feedback */}
            {import.meta.env.DEV && <Agentation endpoint="http://localhost:4747" />}
        </div>
    )
}
