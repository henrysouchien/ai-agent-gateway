import React from 'react';
import { Canvas, MetricCard, Stack, fmtPercent } from '@hank/canvas-kit';
const DATA = { risk: 0.42 };
export default function ValidCanvas(){ return <Canvas title="Risk snapshot" generatedAt="2026-07-21"><Stack><MetricCard label="Risk" value={fmtPercent(DATA.risk)} /></Stack></Canvas>; }
