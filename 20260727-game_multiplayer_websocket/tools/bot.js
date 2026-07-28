// The stand-in for a player during a battle. The only live input is the ammunition toggle, so this
// is the whole of it.
//
// Grape only pays while there is a crew left to break and guns for them to leave silent. Once the
// enemy deck is quiet, or its crew is too deep to shoot away, round shot is what sinks a ship: the
// win goes to whoever takes the helm, and grape barely scratches timber. An earlier version keyed
// off raw damage numbers and quietly went all-grape for any ship with light guns, which meant the
// swivel archetype spent every battle doing no structural damage at all. Re-check this after
// changing damage values in data/parts.js.

// How often the bot looks up, in seconds. It used to decide every tick, which is sixty times a
// second and not a thing a player can do -- switching costs a reload, so the real decision is made
// a handful of times a battle. It was also 13% of the time every harness spent.
//
// It doubles as the step size for playBattle, which has no reason to re-enter the simulation sixty
// times a second when its input only changes this often. battle.advance breaks any dt into fixed
// ticks internally, so the battle is identical either way. Kept at or below the 0.25s that advance()
// will accept in one call.
export const REACTION = 0.25;

export function chooseAmmo(me, enemy) {
  let manned = 0;
  for (const gun of enemy.guns) if (gun.cell.alive && gun.manned) manned++;
  if (enemy.crew <= 0 || manned === 0) return 'round';
  let best = 0;
  for (const gun of me.guns) if (gun.spec.round.damage > best) best = gun.spec.round.damage;
  return enemy.crew <= 6 || best <= 2 ? 'grape' : 'round';
}

// Drives both sides. `mode` is 'grape' for the normal bot or 'round' to pin both to round shot,
// which is how tools/ablate.js measures what the live decision is worth.
export function makeBot(battle, mode = 'grape') {
  const [a, b] = battle.ships;
  let due = 0;

  return {
    update(dt) {
      due -= dt;
      if (due > 0) return;
      due = REACTION;
      if (mode === 'round') {
        battle.setAmmo(0, 'round');
        battle.setAmmo(1, 'round');
        return;
      }
      battle.setAmmo(0, chooseAmmo(a, b));
      battle.setAmmo(1, chooseAmmo(b, a));
    },
  };
}
