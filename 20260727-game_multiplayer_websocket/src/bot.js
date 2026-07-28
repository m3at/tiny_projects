// The stand-in for a player during a battle. The only live input is the ammunition toggle, so this
// is the whole of it. It pairs with autobuild.js, which is the stand-in for the build phase, and
// lives here for the same reason: both the headless harnesses and the in-browser dev autoplay need
// it. While it sat in tools/ the browser had no player at all, so an autoplayed match never once
// switched to grape -- which is the game's only live decision and 15% of its outcomes.
//
// Outside sim/, because it is an input source rather than part of the deterministic core.
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

// Drives every side, or only the seats it is given.
//
//   mode   'grape' for the normal bot, or 'round' to pin everyone to round shot, which is how
//          tools/ablate.js measures what the live decision is worth.
//   apply  how to deliver the choice. Defaults straight to the simulation; the game passes its own,
//          so the ammunition buttons light up as the bot works them.
//   seats  which ship indices to drive, or all of them. Online play fills the empty seats of a
//          short-handed room with bots and leaves the humans alone.
export function makeBot(battle, { mode = 'grape', apply, seats = null } = {}) {
  const set = apply || ((index, ammo) => battle.setAmmo(index, ammo));
  let due = 0;

  return {
    update(dt) {
      due -= dt;
      if (due > 0) return;
      due = REACTION;
      const ships = battle.ships;
      for (let i = 0; i < ships.length; i++) {
        const ship = ships[i];
        if (ship.out || (seats !== null && !seats.includes(ship.index))) continue;
        if (mode === 'round') {
          set(ship.index, 'round');
          continue;
        }
        // Against whoever it is actually fighting. In a melee the ship you are pounding is not
        // necessarily the one whose crew is thin, and the gun crews answer to the target they have.
        const foe = ship.target !== null && !ship.target.out ? ship.target : null;
        set(ship.index, foe === null ? 'round' : chooseAmmo(ship, foe));
      }
    },
  };
}
