// One-command rendering check: frame tails, layer cost, and JavaScript profile.
//
//   node tools/render.js              8s samples, four ships, 1280x720 fill test
//   node tools/render.js 12 2 1920 1080
//
// Run ./tools/dev.sh first. The focused tools remain useful individually; this is the repeatable
// setup for a rendering pass, keeping their inputs together so before/after runs are comparable.

import { spawn } from 'node:child_process';

const seconds = Math.max(2, Number(process.argv[2] || 8));
const players = Math.max(2, Math.min(4, Number(process.argv[3] || 4)));
const width = Math.max(320, Number(process.argv[4] || 1280));
const height = Math.max(240, Number(process.argv[5] || 720));

const checks = [
  ['frame tails', ['tools/frames.js', String(seconds), String(players)]],
  ['layer cost', ['tools/fill.js', String(width), String(height)]],
  ['JavaScript profile', ['tools/profile.js', String(seconds), 'battle']],
];

for (const [label, args] of checks) {
  console.log(`\n=== ${label} ===`);
  const code = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, { stdio: 'inherit' });
    child.once('error', reject);
    child.once('exit', (status, signal) => resolve(signal ? 1 : (status ?? 1)));
  });
  if (code !== 0) process.exit(code);
}
