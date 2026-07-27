// Every colour the renderer uses. Part colours stay in data/parts.js because they are part
// identity rather than styling; these are the surroundings the parts sit in.
//
// Keep the two PLAYER entries in step with --p1 and --p2 in styles.css.

export const PLAYER = [
  { deck: 0x2b3a46, hull: 0x3d5568, flag: 0x5fa8ff },
  { deck: 0x453029, hull: 0x60403a, flag: 0xff7a5f },
];

export const SEA = {
  background: 0x0a1017,
  water: 0x15384c,
  arenaRing: 0x2c5670,
  windStreak: 0x6f9fb8,
  spar: 0xcabfa6, // masts, flagpoles, other bare timber
};

export const LIGHT = {
  sky: 0xbcd8f0,
  ground: 0x2a2418,
  sun: 0xfff0d8,
};

// A destroyed cell leaves a hole; this is what you see through it.
export const HOLE = 0x101820;

export const FX = {
  muzzleFlash: 0xffd9a0,
  muzzleSmoke: 0x9fb0bd,
  impactRound: 0xffb887,
  impactGrape: 0xffe6a8,
  impactRing: 0xffc79b,
  crew: 0xff6a6a,
  debris: 0x6b5a4a,
  splinters: 0x8a7250,
  destroyRing: 0xd8b25c,
  blastCore: 0xfff2c8,
  blastFire: 0xff9840,
  blastRing: 0xffd9a0,
  blastSmoke: 0x5a4a3c,
  splash: 0x9fd0e8,
  roundShot: 0x22262b,
  grapeShot: 0xc8b48a,
  ghost: 0x445f33, // build-phase hover highlight
};
