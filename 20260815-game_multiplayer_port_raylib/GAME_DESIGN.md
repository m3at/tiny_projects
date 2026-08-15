# Broadside game design

Broadside is a build-then-watch game for two to four ships. Human captains share one keyboard and
take complete shipyard turns in seat order; bot seats build and lock immediately. A round has four
beats:

1. award scrap and move every surviving design onto the new hull;
2. fit, remove, reroll, and refit parts during the active captain's authority-owned timer;
3. watch an autonomous battle while choosing round or grape shot;
4. score the placing and carry surviving damage into the next round.

The first captain to three wins. The match lasts at most five rounds; after the final round a unique
score leader wins, otherwise the match is a draw. Lower placings receive progressively more comeback
scrap, so a four-ship loss is not treated as one undifferentiated result.

## Shipyard decisions

The centre spine carries the free fixed helm and supports crew quarters, masts, magazines, and other
non-broadside equipment. Flank cells carry broadside guns. Bow-only weapons must be placed in the bow
rows. Empty cells are real holes: a shot can pass through an empty outer row and reach the spine.
Timber fills holes cheaply; heavy timber trades cost for damage soak.

Every offer contains timber, a powder magazine, and crew quarters plus two distinct random part
types. A reroll costs two scrap. A part bought during the current shipyard phase refunds its full cost
when removed; older equipment does not. Refit repairs the worst surviving damage first and spends
half-cost per repaired part until the purse cannot fund another repair.

Auto-fit is an ergonomic shortcut, not a second builder or a rules bypass. It chooses from the current
authority-generated offer and sends ordinary typed placement commands for magazine, crew, mast, guns,
and timber. Every cost and placement is still validated by the authority.

Crew quarters supply hands. Non-gun stations are staffed before guns, and guns use a stable ordering
that favors the broadside expected to engage first. A ship without a live magazine cannot fire. Too
few hands leave later guns unmanned; excess masts beyond the hull's useful count add no speed.

## Battle decisions

Ships approach their effective weapon range, turn to keep batteries bearing, and share an orbit
sense so the fleet circles rather than sailing away in parallel. Wind changes movement speed but not
build legality. Broadside arcs are intentionally one-sided, so choosing a flank is a meaningful bet.

Round shot favors structural damage. Grape trades structural power for crew loss and can unman guns.
Changing ammunition adds a reload penalty to the current battery. Projectiles can pass through holes,
destroy cells, detonate magazines, sever sections disconnected from the helm, bring down masts, and
force a ship to strike when its helm is lost.

Three- and four-ship battles begin with round-robin targets. Live ships reconsider every 0.6 seconds
and switch only when another foe is within the 0.8 distance margin. Collision spacing uses oriented
hull support radii; grinding damage scales with relative speed. A battle ends with one survivor, a
stalemate, or the 40-second bell. Bell decisions use remaining structure and a draw margin. Placings
are always best-first.

## Why the rules are protected

All randomness comes from a seeded generator. Simulation advances at fixed 60 Hz ticks, collapses
only trigonometric results to float32, and disables fused contraction. Ammunition choices are stamped
by tick and replayed. Every client retains battle initialization and the complete input log so late
inputs, reconnects, and checksum disagreement can rebuild from tick zero. The authority alone awards
economy and verdicts.

Broadside arcs, shared orbit sense, holes, fixed ticks, stable manning, hull damage pacing, magazine
chains, and the melee target margin are load-bearing. Change them only with the native golden,
900-battle semantic fixture, melee envelope, and network replay checks.

## Presentation contract

Presentation may explain simulation state but must not create it. Damage tint, missing deck cells,
falling masts, sinking, muzzle flashes, splinters, foam, hit flashes, logs, and sound are derived from
typed battle effects and runtime state. Adaptive rendering may reduce world resolution and effect
density; it must never change simulation ticks, input timing, authority decisions, or native-resolution
UI.

The world target uses adaptive supersampling, multisampling, and post-process edge smoothing while UI
remains at native window resolution. Ship geometry shares one local heading frame, and
presentation-owned sinking continues during a frozen verdict without advancing simulation. New
battles clear camera, shake, particle, and wreck history atomically.
