// Shared bits of the headless harnesses. Deliberately dependency-free of src/ so that
// tools/ablate.js can apply them to patched copies of the source tree.

export const pct = (n) => `${(n * 100).toFixed(0)}%`;

export function budgetFor(rounds, hullIndex) {
  return rounds.slice(0, hullIndex + 1).reduce((sum, r) => sum + r.scrap, 0);
}

// The stand-in for a player's ammunition decision, used everywhere a bot needs one.
// Grape is worth it when our round shot would only bounce off their timbers, or when their
// crew is already thin enough to break. A real player learns roughly this.
export function chooseAmmo(myGuns, enemyCrew) {
  const best = myGuns.reduce((m, g) => Math.max(m, g.spec.round.damage), 0);
  return enemyCrew > 0 && (best <= 5 || enemyCrew <= 6) ? 'grape' : 'round';
}

export function applyBotAmmo(battle, me, enemy) {
  battle.setAmmo(me.index, chooseAmmo(me.guns, enemy.crew));
}

// Run a battle to its conclusion with bots working the ammunition. `mods` supplies the
// modules so a patched source tree can be substituted.
export function playBattle({ ship, battle: battleMod, config }, designs, hullIndex, seed, opts = {}) {
  const battle = battleMod.createBattle({
    designs,
    hullIndex,
    seed,
    windTo: opts.windTo ?? (seed % 360) * (Math.PI / 180),
  });
  let guard = 0;
  while (!battle.over && guard++ < 60 / config.TICK) {
    if (opts.grape === false) {
      battle.setAmmo(0, 'round');
      battle.setAmmo(1, 'round');
    } else {
      applyBotAmmo(battle, battle.ships[0], battle.ships[1]);
      applyBotAmmo(battle, battle.ships[1], battle.ships[0]);
    }
    battle.advance(config.TICK);
  }
  return {
    battle,
    winner: battle.winner,
    time: battle.time,
    reason: battle.reason,
    decisive: battle.time < config.BATTLE_CAP - 0.1,
    struct: [ship.structureFraction(battle.ships[0]), ship.structureFraction(battle.ships[1])],
  };
}
