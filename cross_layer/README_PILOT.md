# Cross-layer pilot (G2 / G3 / G4)

This package upgrades Stage 0 into the first fast end-to-end pilot.

## Groups

- **G2**: simple stacking — BA (`R=4`) + fixed RandAugment `M=5` + random Mixup/CutMix switch.
- **G3**: fixed A/B/C/D roles + shuffled `mixing degree <-> M` correspondence.
- **G4**: fixed A/B/C/D roles + inverse coupling (`stronger mixing -> weaker RA`).

The pilot deliberately does **not** evaluate on CIFAR-100 test by default.
Model selection is based on a fixed 45K train / 5K validation split.

## Budget

Default pilot:

```text
source batch = 32
R            = 4
model inputs = 128 / update
updates      = 10,000
```

This is a mechanism pilot, not the final 70,400-update experiment.

## Install

Copy / overwrite the `cross_layer/` directory at the repository root:

```text
CutMix-PyTorch/
  train.py
  model/
  cross_layer/
```

Also copy `run_pilot.sh` to the repository root.

## One-command pilot

```bash
bash run_pilot.sh
```

It runs G2 -> G3 -> G4 and then prints a comparison table.

For an even faster first signal:

```bash
UPDATES=5000 bash run_pilot.sh
```

For the default 10K pilot:

```bash
UPDATES=10000 bash run_pilot.sh
```

## Most important sanity signal

Across many updates:

```text
G3 mean corr(s, M) ~= 0
G4 mean corr(s, M) << 0
```

A single 32-source batch can have a non-zero random G3 correlation by chance.
The running average is the relevant control.

## Core pilot result

The key line printed at the end is:

```text
CORE PILOT: G4 - G3 = ... pp
```

Treat this only as a single-seed validation signal.
If G4 consistently looks promising, move to:

1. full 70,400 updates,
2. G3/G4/G5,
3. >= 3 paired seeds,
4. best-validation checkpoint -> test once.
