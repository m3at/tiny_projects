// Every colour the renderer uses. Part colours stay in data/parts.js because they are part
// identity rather than styling; these are the surroundings the parts sit in.
//
// Keep the four PLAYER entries in step with --p1 to --p4 in styles.css.

// `spine` tints the centre column a shade lighter than the flanks, so the rule that
// broadsides go on the flanks and vitals down the spine is visible on the deck itself.
//
// Four hulls have to be told apart at a glance on dark water while every part aboard them keeps its
// own colour, so the four are separated by hue and not by lightness: blue, rust, sea-green, violet.
// The decks are the same colour at a tenth of the saturation, which is what keeps a gun deck reading
// as a gun deck on all four ships.
export const PLAYER = [
  { deck: 0x25333d, spine: 0x334654, hull: 0x3d5568, flag: 0x5fa8ff },
  { deck: 0x3d2a24, spine: 0x513a30, hull: 0x60403a, flag: 0xff7a5f },
  { deck: 0x243a31, spine: 0x30503f, hull: 0x3a604d, flag: 0x63d1a8 },
  { deck: 0x33283d, spine: 0x453655, hull: 0x543f66, flag: 0xc98cf0 },
];

export const SEA = {
  // What `water` used to render as once the hemisphere and sun lights were applied to a flat
  // upward-facing plane. Under an orthographic camera both the normal and the view direction are
  // constant across that plane, so every one of its pixels came out the same colour -- measured,
  // not guessed: nine samples across the frame all read 172d3a. It is the clear colour now, and
  // there is no sea mesh. See the rendering notes in CLAUDE.md.
  water: 0x172d3a,
  swell: 0x234353,
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
  foam: 0xb8deea,
  wake: 0x6c9caf,
  roundShot: 0x22262b,
  grapeShot: 0xc8b48a,
  ghost: 0x445f33, // build-phase hover highlight
  arc: 0xd8b25c, // firing-arc preview
};
