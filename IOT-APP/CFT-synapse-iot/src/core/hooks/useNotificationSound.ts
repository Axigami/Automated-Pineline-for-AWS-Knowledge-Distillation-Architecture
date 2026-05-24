/**
 * Notification Sound Controller Hook
 * 
 * Manages notification sound playback with 3-second debouncing
 * to prevent repeated sounds when multiple alerts arrive.
 * 
 * Uses Web Audio API to generate a "glassy bell" sound (same as Live Monitor)
 */

import { useRef, useCallback, useEffect } from 'react';

interface NotificationSoundConfig {
  debounceMs: number;
}

const DEFAULT_CONFIG: NotificationSoundConfig = {
  debounceMs: 3000, // 3 seconds
};

let audioCtx: AudioContext | null = null;

/**
 * Play alert sound using Web Audio API
 * Creates a "glassy bell" sound with two oscillators
 */
const playAlertSound = () => {
  try {
    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioContextClass) return;
      audioCtx = new AudioContextClass();
    }

    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    
    // Create 2 layers of sound for a "Glassy Bell" effect
    const osc1 = audioCtx.createOscillator();
    const osc2 = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    osc1.connect(gainNode);
    osc2.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    // Layer 1: Base tone (Sine wave) - smooth and deep
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(1046.50, audioCtx.currentTime); // Note C6

    // Layer 2: Harmonic (Triangle wave) - creates the bell's shimmer
    osc2.type = 'triangle';
    osc2.frequency.setValueAtTime(2093.00, audioCtx.currentTime); // Note C7 (Octave)

    // "Bell strike" effect: Sharp attack and gradual decay
    gainNode.gain.setValueAtTime(0, audioCtx.currentTime);
    gainNode.gain.linearRampToValueAtTime(0.4, audioCtx.currentTime + 0.02);
    gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.8);

    osc1.start(audioCtx.currentTime);
    osc2.start(audioCtx.currentTime);
    osc1.stop(audioCtx.currentTime + 0.8);
    osc2.stop(audioCtx.currentTime + 0.8);

  } catch (err) {
    console.error('[NotificationSound] Audio playback error:', err);
  }
};

export function useNotificationSound(config: Partial<NotificationSoundConfig> = {}) {
  const finalConfig = { ...DEFAULT_CONFIG, ...config };
  
  const lastPlayTimeRef = useRef<number>(0);
  const enabledRef = useRef<boolean>(true);

  // Initialize - load preference from localStorage
  useEffect(() => {
    const stored = localStorage.getItem('notification_sound_enabled');
    enabledRef.current = stored === null ? true : stored === 'true';
  }, []);

  /**
   * Play notification sound with debouncing
   * Only plays if:
   * 1. Sounds are enabled
   * 2. At least 3 seconds have passed since last play
   */
  const playSound = useCallback(() => {
    if (!enabledRef.current) {
      console.log('[NotificationSound] Sound disabled by user');
      return;
    }

    const now = Date.now();
    const timeSinceLastPlay = now - lastPlayTimeRef.current;

    if (timeSinceLastPlay < finalConfig.debounceMs) {
      console.log(`[NotificationSound] Debounced - ${timeSinceLastPlay}ms since last play (need ${finalConfig.debounceMs}ms)`);
      return;
    }

    // Play sound
    lastPlayTimeRef.current = now;
    playAlertSound();
    console.log('[NotificationSound] Sound played');
  }, [finalConfig.debounceMs]);

  /**
   * Enable or disable notification sounds
   */
  const setEnabled = useCallback((enabled: boolean) => {
    enabledRef.current = enabled;
    localStorage.setItem('notification_sound_enabled', String(enabled));
    console.log(`[NotificationSound] ${enabled ? 'Enabled' : 'Disabled'}`);
  }, []);

  /**
   * Check if notification sounds are enabled
   */
  const isEnabled = useCallback(() => {
    return enabledRef.current;
  }, []);

  return {
    playSound,
    setEnabled,
    isEnabled,
  };
}
