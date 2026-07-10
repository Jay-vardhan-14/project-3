import { useEffect, useRef, useState } from 'react';

export interface PollState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

// Fetch on mount and re-fetch every `interval` ms for a real-time feel.
export function usePoll<T>(fetcher: () => Promise<T>, interval = 30000): PollState<T> {
  const [state, setState] = useState<PollState<T>>({ data: null, error: null, loading: true });
  const ref = useRef(fetcher);
  ref.current = fetcher;

  useEffect(() => {
    let alive = true;
    const run = async () => {
      try {
        const data = await ref.current();
        if (alive) setState({ data, error: null, loading: false });
      } catch (err) {
        if (alive) {
          setState((prev) => ({
            ...prev,
            error: err instanceof Error ? err.message : 'Failed to load',
            loading: false,
          }));
        }
      }
    };
    run();
    const id = setInterval(run, interval);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [interval]);

  return state;
}
