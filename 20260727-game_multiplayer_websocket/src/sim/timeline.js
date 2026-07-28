// A battle plus its input stream, advanced by tick number.
//
// Every input in this game is an ammunition toggle, and the only thing two machines can agree on
// about when it happened is a tick number. This is the one piece of code that turns a stamped input
// stream into a battle, and both the authority and every client replica use it -- if the server
// applied inputs one way and a client another, the two would diverge for a reason no checksum could
// explain.
//
// It lives in sim/ because it is part of the deterministic core, even though an input source is not:
// bot.js sits outside for that reason and this does not. Given the same battle parameters and the
// same set of inputs, runTo(n) produces the same battle on any machine, whatever order the inputs
// arrived in and however many calls it took to get there.

// Inputs are kept sorted by tick, then by seat, so the order two toggles are applied in does not
// depend on which packet the server happened to read first.
function before(a, b) {
  if (a.tick !== b.tick) return a.tick < b.tick;
  return a.seat < b.seat;
}

export function createTimeline(battle) {
  const inputs = [];
  let cursor = 0;

  return {
    battle,
    inputs,

    // Returns false for a duplicate, so a re-sent input is not applied twice.
    add(input) {
      for (const existing of inputs) {
        if (existing.tick === input.tick && existing.seat === input.seat) return false;
      }
      let at = inputs.length;
      while (at > 0 && before(input, inputs[at - 1])) at--;
      inputs.splice(at, 0, input);
      // An input inserted behind the cursor has already been passed over: the caller is replaying
      // history, which only happens on a resync, and resync() rewinds the cursor itself.
      if (at < cursor) cursor = at;
      return true;
    },

    // Advance to `tick`, applying anything stamped for a tick as it comes due. An input stamped for
    // a tick already run is applied at the next boundary instead of being dropped -- late, and
    // therefore a divergence from a machine that had it on time, which is what the checksum is for.
    runTo(tick) {
      while (battle.tickCount < tick && !battle.over) {
        while (cursor < inputs.length && inputs[cursor].tick <= battle.tickCount) {
          const input = inputs[cursor++];
          battle.setAmmo(input.seat, input.ammo);
        }
        battle.advanceTicks(1);
      }
      // Anything stamped for a tick at or before where we stopped, once the battle is over, would
      // otherwise sit in front of the cursor for ever.
      while (cursor < inputs.length && inputs[cursor].tick < battle.tickCount) cursor++;
    },

    // Advance to `target`, stopping exactly on every multiple of `every` to hand the battle to
    // `onMark`. The authority fingerprints the state at those ticks and the replica does the same,
    // and the reason it is one function rather than two loops is that a checksum taken at tick 61 on
    // one machine and tick 60 on the other compares nothing at all. Overshooting is the natural
    // thing for both sides to do -- each advances by however much wall time has passed -- so neither
    // is allowed to.
    runToMarks(target, every, onMark) {
      while (battle.tickCount < target && !battle.over) {
        const nextMark = (Math.floor(battle.tickCount / every) + 1) * every;
        const before = battle.tickCount;
        this.runTo(Math.min(target, nextMark));
        if (battle.tickCount === before) break; // over, or nothing to do
        if (battle.tickCount % every === 0) onMark(battle.tickCount, battle);
      }
    },

    rewind() {
      cursor = 0;
    },
  };
}
