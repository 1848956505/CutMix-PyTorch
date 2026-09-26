from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torchvision import transforms


CouplingMode = Literal["shuffle", "inverse", "positive"]


@dataclass
class BatchPlan:
    mix_partner: torch.Tensor
    cut_partner: torch.Tensor

    lam_mix: torch.Tensor
    lam_cut_actual: torch.Tensor
    cut_boxes: torch.Tensor

    mix_degree_mix: torch.Tensor
    mix_degree_cut: torch.Tensor

    m_inverse_mix: torch.Tensor
    m_inverse_cut: torch.Tensor
    m_shuffle_mix: torch.Tensor
    m_shuffle_cut: torch.Tensor
    m_positive_mix: torch.Tensor
    m_positive_cut: torch.Tensor


def derangement(n: int, generator: torch.Generator) -> torch.Tensor:
    """Permutation with no fixed points."""
    if n < 2:
        raise ValueError("A derangement requires n >= 2.")

    base = torch.arange(n)

    for _ in range(128):
        perm = torch.randperm(n, generator=generator)
        if not torch.any(perm == base):
            return perm

    shift = int(torch.randint(1, n, (1,), generator=generator).item())
    return torch.roll(base, shifts=shift)


def mixing_degree(lam: torch.Tensor) -> torch.Tensor:
    """Symmetric mixing-balance proxy in [0, 1]."""
    return 4.0 * lam * (1.0 - lam)


def inverse_magnitude(
    lam: torch.Tensor,
    m_min: int = 3,
    m_max: int = 9,
) -> tuple[torch.Tensor, torch.Tensor]:
    s = mixing_degree(lam)
    m = torch.round(
        m_min + (m_max - m_min) * (1.0 - s)
    ).to(torch.int64)
    return s, m.clamp_(m_min, m_max)


def positive_assignment(
    s: torch.Tensor,
    magnitudes: torch.Tensor,
) -> torch.Tensor:
    """Same M multiset, but larger s gets larger M."""
    order_s = torch.argsort(s)
    sorted_m = torch.sort(magnitudes).values
    out = torch.empty_like(magnitudes)
    out[order_s] = sorted_m
    return out


def sample_cutmix_box(
    lam: float,
    height: int,
    width: int,
    generator: torch.Generator,
) -> tuple[tuple[int, int, int, int], float]:
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = int(torch.randint(0, width, (1,), generator=generator).item())
    cy = int(torch.randint(0, height, (1,), generator=generator).item())

    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, height)

    actual_area = (x2 - x1) * (y2 - y1)
    lam_actual = 1.0 - actual_area / float(height * width)

    return (x1, y1, x2, y2), lam_actual


def make_plan(
    batch_size: int,
    height: int,
    width: int,
    *,
    seed: int,
    m_min: int = 3,
    m_max: int = 9,
) -> BatchPlan:
    """Shared plan for G3/G4/G5.

    With the same seed, separate runs receive exactly the same partner
    permutations, lambda values, CutMix boxes, and M multisets.
    """
    g = torch.Generator().manual_seed(seed)

    mix_partner = derangement(batch_size, g)
    cut_partner = derangement(batch_size, g)

    # Beta(1, 1) == Uniform(0, 1).
    lam_mix = torch.rand(batch_size, generator=g)
    lam_cut_sampled = torch.rand(batch_size, generator=g)

    boxes = []
    lam_cut_actual = []

    for lam in lam_cut_sampled.tolist():
        box, actual = sample_cutmix_box(
            lam, height, width, generator=g
        )
        boxes.append(box)
        lam_cut_actual.append(actual)

    cut_boxes = torch.tensor(boxes, dtype=torch.int64)
    lam_cut_actual = torch.tensor(lam_cut_actual, dtype=torch.float32)

    s_mix, m_inverse_mix = inverse_magnitude(
        lam_mix, m_min=m_min, m_max=m_max
    )
    s_cut, m_inverse_cut = inverse_magnitude(
        lam_cut_actual, m_min=m_min, m_max=m_max
    )

    m_shuffle_mix = m_inverse_mix[torch.randperm(batch_size, generator=g)]
    m_shuffle_cut = m_inverse_cut[torch.randperm(batch_size, generator=g)]

    m_positive_mix = positive_assignment(s_mix, m_inverse_mix)
    m_positive_cut = positive_assignment(s_cut, m_inverse_cut)

    return BatchPlan(
        mix_partner=mix_partner,
        cut_partner=cut_partner,
        lam_mix=lam_mix,
        lam_cut_actual=lam_cut_actual,
        cut_boxes=cut_boxes,
        mix_degree_mix=s_mix,
        mix_degree_cut=s_cut,
        m_inverse_mix=m_inverse_mix,
        m_inverse_cut=m_inverse_cut,
        m_shuffle_mix=m_shuffle_mix,
        m_shuffle_cut=m_shuffle_cut,
        m_positive_mix=m_positive_mix,
        m_positive_cut=m_positive_cut,
    )


class CrossLayerBatchBuilder:
    """Build Stage-0/Stage-1 batches from raw uint8 CIFAR images."""

    ROLE_CODE = {"A": 11, "B": 23, "C": 37, "D": 53, "G2": 71}

    def __init__(
        self,
        *,
        num_classes: int = 100,
        ra_n: int = 3,
        role_b_m: int = 9,
        fixed_m: int = 5,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        fill=(125, 123, 114),
        base_seed: int = 20170922,
    ):
        self.num_classes = num_classes
        self.ra_n = ra_n
        self.role_b_m = role_b_m
        self.fixed_m = fixed_m
        self.base_seed = base_seed

        self.base_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

        self.ra_cache = {
            m: transforms.RandAugment(
                num_ops=ra_n,
                magnitude=m,
                num_magnitude_bins=31,
                fill=fill,
            )
            for m in range(31)
        }

        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def _seed_for(
        self,
        role: str,
        position: int,
        *,
        step: int,
        partner: bool = False,
        replica: int = 0,
    ) -> int:
        # Deterministic across separate G3/G4 runs while changing every step.
        return int(
            self.base_seed
            + step * 1_000_003
            + self.ROLE_CODE[role] * 100_000
            + position * 97
            + replica * 7_001
            + (7_919 if partner else 0)
        )

    def _augment_source(
        self,
        image: torch.Tensor,
        *,
        magnitude: int,
        seed: int,
        use_ra: bool,
    ) -> torch.Tensor:
        if image.dtype != torch.uint8:
            raise TypeError(
                f"Expected uint8 source image, got {image.dtype}"
            )

        # Keep caller RNG untouched so separate experiment groups stay paired.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)

            x = self.base_transform(image)

            if use_ra:
                x = self.ra_cache[int(magnitude)](x)

        return x.to(torch.float32) / 255.0

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    @staticmethod
    def _m_for(
        plan: BatchPlan,
        mode: CouplingMode,
        branch: Literal["mix", "cut"],
    ) -> torch.Tensor:
        if mode == "inverse":
            return getattr(plan, f"m_inverse_{branch}")
        if mode == "shuffle":
            return getattr(plan, f"m_shuffle_{branch}")
        if mode == "positive":
            return getattr(plan, f"m_positive_{branch}")
        raise ValueError(f"Unknown coupling mode: {mode}")

    def build_roles(
        self,
        raw_images: torch.Tensor,
        labels: torch.Tensor,
        plan: BatchPlan,
        *,
        mode: CouplingMode,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Build G3/G4/G5: A/B/C/D fixed roles."""
        batch_size = raw_images.size(0)

        if batch_size != plan.mix_partner.numel():
            raise ValueError("Plan batch size does not match source batch.")

        one_hot = F.one_hot(
            labels.to(torch.int64),
            num_classes=self.num_classes,
        ).to(torch.float32)

        images = []
        targets = []
        roles = []
        source_ids = []

        # A: Base.
        for i in range(batch_size):
            x = self._augment_source(
                raw_images[i],
                magnitude=0,
                seed=self._seed_for("A", i, step=step),
                use_ra=False,
            )
            images.append(self._normalize(x))
            targets.append(one_hot[i])
            roles.append("A")
            source_ids.append(i)

        # B: Strong single-source RA.
        for i in range(batch_size):
            x = self._augment_source(
                raw_images[i],
                magnitude=self.role_b_m,
                seed=self._seed_for("B", i, step=step),
                use_ra=True,
            )
            images.append(self._normalize(x))
            targets.append(one_hot[i])
            roles.append("B")
            source_ids.append(i)

        # C: Mixup.
        m_mix = self._m_for(plan, mode, "mix")

        for i in range(batch_size):
            j = int(plan.mix_partner[i].item())
            lam = float(plan.lam_mix[i].item())
            magnitude = int(m_mix[i].item())

            xa = self._augment_source(
                raw_images[i],
                magnitude=magnitude,
                seed=self._seed_for(
                    "C", i, step=step, partner=False
                ),
                use_ra=True,
            )
            xb = self._augment_source(
                raw_images[j],
                magnitude=magnitude,
                seed=self._seed_for(
                    "C", i, step=step, partner=True
                ),
                use_ra=True,
            )

            mixed = lam * xa + (1.0 - lam) * xb
            target = lam * one_hot[i] + (1.0 - lam) * one_hot[j]

            images.append(self._normalize(mixed))
            targets.append(target)
            roles.append("C")
            source_ids.append(i)

        # D: CutMix.
        m_cut = self._m_for(plan, mode, "cut")

        for i in range(batch_size):
            j = int(plan.cut_partner[i].item())
            lam_actual = float(plan.lam_cut_actual[i].item())
            magnitude = int(m_cut[i].item())

            xa = self._augment_source(
                raw_images[i],
                magnitude=magnitude,
                seed=self._seed_for(
                    "D", i, step=step, partner=False
                ),
                use_ra=True,
            )
            xb = self._augment_source(
                raw_images[j],
                magnitude=magnitude,
                seed=self._seed_for(
                    "D", i, step=step, partner=True
                ),
                use_ra=True,
            )

            x1, y1, x2, y2 = map(int, plan.cut_boxes[i].tolist())

            mixed = xa.clone()
            mixed[:, y1:y2, x1:x2] = xb[:, y1:y2, x1:x2]

            target = (
                lam_actual * one_hot[i]
                + (1.0 - lam_actual) * one_hot[j]
            )

            images.append(self._normalize(mixed))
            targets.append(target)
            roles.append("D")
            source_ids.append(i)

        metadata = {
            "roles": roles,
            "source_ids": torch.tensor(source_ids, dtype=torch.int64),
            "mix_partner": plan.mix_partner.clone(),
            "cut_partner": plan.cut_partner.clone(),
            "lam_mix": plan.lam_mix.clone(),
            "lam_cut_actual": plan.lam_cut_actual.clone(),
            "mix_degree_mix": plan.mix_degree_mix.clone(),
            "mix_degree_cut": plan.mix_degree_cut.clone(),
            "m_mix": m_mix.clone(),
            "m_cut": m_cut.clone(),
            "mode": mode,
        }

        return (
            torch.stack(images, dim=0),
            torch.stack(targets, dim=0),
            metadata,
        )

    def build_simple_stack(
        self,
        raw_images: torch.Tensor,
        labels: torch.Tensor,
        *,
        step: int,
        repeats: int = 4,
        switch_prob: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Build G2: BA + fixed RA(M=5) + random Mixup/CutMix switch.

        Each source gets `repeats` independently planned mixed views.
        No fixed A/B/C/D role assignment is imposed.
        """
        batch_size = raw_images.size(0)
        height, width = raw_images.shape[-2:]

        one_hot = F.one_hot(
            labels.to(torch.int64),
            num_classes=self.num_classes,
        ).to(torch.float32)

        images = []
        targets = []
        op_names = []
        source_ids = []
        self_pair_count = 0

        for r in range(repeats):
            g = torch.Generator().manual_seed(
                self.base_seed + step * 1_000_003 + r * 13_337 + 17
            )
            partner = derangement(batch_size, g)
            lam_all = torch.rand(batch_size, generator=g)
            use_cutmix = torch.rand(batch_size, generator=g) < switch_prob

            for i in range(batch_size):
                j = int(partner[i].item())
                if j == i:
                    self_pair_count += 1

                lam = float(lam_all[i].item())
                magnitude = self.fixed_m

                xa = self._augment_source(
                    raw_images[i],
                    magnitude=magnitude,
                    seed=self._seed_for(
                        "G2", i, step=step, partner=False, replica=r
                    ),
                    use_ra=True,
                )
                xb = self._augment_source(
                    raw_images[j],
                    magnitude=magnitude,
                    seed=self._seed_for(
                        "G2", i, step=step, partner=True, replica=r
                    ),
                    use_ra=True,
                )

                if bool(use_cutmix[i].item()):
                    box, lam_actual = sample_cutmix_box(
                        lam, height, width, generator=g
                    )
                    x1, y1, x2, y2 = box

                    mixed = xa.clone()
                    mixed[:, y1:y2, x1:x2] = xb[:, y1:y2, x1:x2]

                    target = (
                        lam_actual * one_hot[i]
                        + (1.0 - lam_actual) * one_hot[j]
                    )
                    op_names.append("cutmix")
                else:
                    mixed = lam * xa + (1.0 - lam) * xb
                    target = lam * one_hot[i] + (1.0 - lam) * one_hot[j]
                    op_names.append("mixup")

                images.append(self._normalize(mixed))
                targets.append(target)
                source_ids.append(i)

        metadata = {
            "ops": op_names,
            "source_ids": torch.tensor(source_ids, dtype=torch.int64),
            "self_pair_count": self_pair_count,
            "mode": "simple_stack",
        }

        return (
            torch.stack(images, dim=0),
            torch.stack(targets, dim=0),
            metadata,
        )
