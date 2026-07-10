import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AlertBanner } from './AlertBanner';
import type { AlertRow } from '../../api/client';

const alert = (over: Partial<AlertRow>): AlertRow => ({
  id: '1',
  alert_type: 'drift_critical',
  severity: 'critical',
  message: 'Critical drift detected: score=1.0000',
  is_resolved: false,
  metadata: null,
  created_at: new Date().toISOString(),
  ...over,
});

describe('AlertBanner', () => {
  it('renders the message when an unresolved alert is active', () => {
    render(<AlertBanner alerts={[alert({})]} />);
    expect(screen.getByText(/Critical drift detected/)).toBeInTheDocument();
  });

  it('renders nothing when all alerts are resolved', () => {
    const { container } = render(<AlertBanner alerts={[alert({ is_resolved: true })]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('prefers a critical alert and counts the rest', () => {
    render(
      <AlertBanner
        alerts={[
          alert({ id: 'a', severity: 'warning', message: 'warn', alert_type: 'drift_warning' }),
          alert({ id: 'b', severity: 'critical', message: 'crit' }),
        ]}
      />,
    );
    expect(screen.getByText('crit')).toBeInTheDocument();
    expect(screen.getByText(/\+1 more/)).toBeInTheDocument();
  });
});
