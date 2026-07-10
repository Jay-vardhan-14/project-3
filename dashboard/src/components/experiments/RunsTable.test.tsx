import { describe, it, expect } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RunsTable } from './RunsTable';
import type { ExperimentRow } from '../../api/client';

const runs: ExperimentRow[] = [
  {
    run_id: 'r-base',
    run_name: 'baseline-run',
    experiment: 'baseline-logreg',
    status: 'FINISHED',
    start_time: Date.now(),
    metrics: { f1_macro: 0.85, accuracy: 0.85, precision_macro: 0.85, recall_macro: 0.85 },
  },
  {
    run_id: 'r-tf',
    run_name: 'distilbert-run',
    experiment: 'distilbert-sentiment',
    status: 'FINISHED',
    start_time: Date.now(),
    metrics: { f1_macro: 0.49, accuracy: 0.55, precision_macro: 0.5, recall_macro: 0.5 },
  },
];

const bodyRows = () => screen.getAllByRole('row').slice(1); // drop header row

describe('RunsTable', () => {
  it('renders both baseline and transformer runs', () => {
    render(<RunsTable runs={runs} />);
    expect(screen.getByText('baseline-run')).toBeInTheDocument();
    expect(screen.getByText('distilbert-run')).toBeInTheDocument();
  });

  it('defaults to F1 descending (best model first)', () => {
    render(<RunsTable runs={runs} />);
    expect(within(bodyRows()[0]).getByText('baseline-run')).toBeInTheDocument();
  });

  it('toggles to ascending when the F1 header is clicked', async () => {
    render(<RunsTable runs={runs} />);
    await userEvent.click(screen.getByRole('button', { name: /F1/ }));
    expect(within(bodyRows()[0]).getByText('distilbert-run')).toBeInTheDocument();
  });
});
