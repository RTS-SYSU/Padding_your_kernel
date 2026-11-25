"""Fixed, source-derived workload profiles from the Experiment 23 study."""

from dataclasses import dataclass
from math import gcd

from .model import (
    L1_SETS,
    L1_WAYS,
    LINE_ELEMENTS,
    MODULO_ASSUMPTIONS,
    SECTOR_ELEMENTS,
    Recommendation,
    RowLayout,
    analyze_adjacent_columns,
)


@dataclass(frozen=True, slots=True)
class Preset:
    name: str
    operator: str
    family: str
    case: str
    size: str
    summary: str
    source: str
    operands: tuple[tuple[str, int, int], ...]
    rule: str
    reason: str
    method: str
    facts: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "operator": self.operator,
            "family": self.family,
            "case": self.case,
            "size": self.size,
            "summary": self.summary,
            "source": self.source,
        }


POLYBENCH = "PolyBench/GPU"
RODINIA = "Rodinia"
STUDY = "Analytical GPU Padding Benefit Model, Experiment 23"

_PROFILES = (
    Preset(
        "atax", "ATAX", POLYBENCH, "n2048", "2048 × 2048",
        "Two launches: A times x, followed by the transpose times the intermediate vector.",
        "experiment_04_atax_hardware/atax.cu", (("A", 2048, 2048),),
        "Adjacent-column reuse",
        "The first launch reuses each lane's matrix tag at consecutive columns with one intervening broadcast. "
        "The transpose launch retains 32-byte row alignment.",
        "adjacent",
    ),
    Preset(
        "bicg", "BiCG", POLYBENCH, "n2048", "2048 × 2048",
        "Separate transpose and row products, launched in that order on one stream.",
        "experiment_06_bicg_feasible/bicg.cu", (("A", 2048, 2048),),
        "Adjacent-column reuse",
        "The q row product has the adjacent-column reuse window and one vector broadcast. "
        "The s transpose product retains 32-byte row alignment.",
        "adjacent",
    ),
    Preset(
        "mvt", "MVT", POLYBENCH, "n2048", "2048 × 2048",
        "Add row and transpose products to two independent starting vectors.",
        "experiment_07_mvt/workload.cu", (("A", 2048, 2048),),
        "Adjacent-column reuse",
        "The row product has same-tag A reuse and one y1 broadcast. Alignment also protects the transpose product's issued sectors.",
        "adjacent",
    ),
    Preset(
        "mvt-4096", "MVT", POLYBENCH, "n4096", "4096 × 4096",
        "The second fixed MVT case uses the same two-launch mapping with twice the logical width.",
        "experiment_22_mvt_reuse/run_mvt.py and experiment_07_mvt/workload.cu", (("A", 4096, 4096),),
        "Adjacent-column reuse",
        "The 4096-float width also cancels against both modeled set periods; the one-broadcast bound still applies.",
        "adjacent",
    ),
    Preset(
        "gemm", "GEMM", POLYBENCH, "default", "M=512, N=512, K=256",
        "Full scalar dense product with an in-place C update and a pitched right operand B.",
        "experiment_08_gemm/workload.cu and spec.py", (("B", 256, 512),),
        "Aligned right-operand stream",
        "A warp reads one different B tag at each reduction step. At the packed pitch, an adjacent A broadcast reuse "
        "sees at most one intervening B tag. The covered window already fits.",
        "compact",
    ),
    Preset(
        "2mm", "2MM", POLYBENCH, "default", "M=384, J=512, K=256, L=384",
        "Two dense matrix products with one intermediate and a shared increment on B and C.",
        "experiment_09_2mm/workload.cu and spec.py", (("B", 256, 512), ("C", 512, 384)),
        "Aligned per-stage right operands",
        "B and C advance a full row per reduction step. At the packed pitch each stage issues one aligned right-hand "
        "tag, and the left-hand broadcast reuses within the local bound.",
        "compact",
    ),
    Preset(
        "syrk", "SYRK", POLYBENCH, "default", "A=384 × 256",
        "Full-square rank-k update with one broadcast and one cross-row burst per reduction step.",
        "experiment_10_syrk/workload.cu and spec.py", (("A", 384, 256),),
        "Signed set distance with one broadcast",
        "At pitch 264, only the first and last lanes can compete for the same matrix set in an adjacent-column "
        "reuse interval. With the broadcast reserve, at most two lines intervene.",
        "first_sector",
        (
            "At the 256-float packed pitch, the L1 reuse interval can contain seven same-set matrix tags.",
            "At the first positive 32-byte-aligned pitch, d × 264 modulo 32S can create a collision only for |d|=31.",
        ),
    ),
    Preset(
        "syr2k", "SYR2K", POLYBENCH, "default", "A and B=384 × 256",
        "Full-square rank-2k update with two cross-row bursts and two broadcasts per reduction step.",
        "experiment_11_syr2k/workload.cu and spec.py", (("A", 384, 256), ("B", 384, 256)),
        "Two-matrix reuse with an odd line pitch",
        "A nine-line pitch makes each 32-lane burst injective in both modeled geometries. "
        "One line from the other burst and two broadcasts fit the L1 reuse budget.",
        "odd_lines",
        (
            "The sufficient congruence is (256 + p) modulo 64 = 32.",
            "The smaller aligned pads 8, 16, and 24 have explicit same-set reuse distances 4, 5, and 5 "
            "in the modeled order with equal base-set phases.",
        ),
    ),
    Preset(
        "3dconv", "3DCONV", POLYBENCH, "default", "64 × 128 × 256",
        "Interior three-dimensional stencil. Only input A has a row pitch.",
        "experiment_12_3dconv/workload.cu and spec.py", (("A", 64 * 128, 256),),
        "Stencil window already fits",
        "At most three input plane tags and one output tag occupy any set in the packed single-warp stencil window. "
        "Aligned padding cannot separate adjacent planes in the ideal L1 model.",
        "compact",
        ("The pitch in lines between two planes is 1024 + 4p, a multiple of 32 when p is a multiple of 8.",),
    ),
    Preset(
        "fdtd", "FDTD-2D", POLYBENCH, "default", "512 × 1024, 12 steps",
        "Thirty-six ordered field-update launches. Ex, Ey, and Hz share one row increment.",
        "experiment_13_fdtd/workload.cu and spec.py",
        tuple((name, 512, 1024) for name in ("ex", "ey", "hz")),
        "Per-stage stencil windows already fit",
        "The Ey, Ex, and Hz single-warp windows need at most three, two, and four different tags per set "
        "at the packed pitch. Cross-launch retention is outside this rule.",
        "compact",
    ),
    Preset(
        "hotspot", "HotSpot", RODINIA, "default", "512 × 512, 8 steps",
        "A 16 × 16 shared-memory tile advances four time steps per launch.",
        "experiment_14_hotspot/workload.cu and spec.py",
        tuple((name, 512, 512) for name in ("temp0", "temp1", "power")),
        "Temporal reuse is in shared memory",
        "The known tile-local temporal reuse occurs in shared memory after loading temperature and power. "
        "The packed global pitch is aligned; no ordered cross-block halo reuse window is specified.",
        "compact",
    ),
    Preset(
        "srad", "SRAD v2", RODINIA, "default", "512 × 512, 4 iterations",
        "The coefficient stage uses a 16 × 16 tile and neighbor reads; six global fields share one increment.",
        "experiment_15_srad/workload.cu and spec.py",
        tuple((name, 512, 512) for name in ("image", "coeff", "north", "south", "west", "east")),
        "Coefficient CTA neighbor reuse",
        "With one way reserved for the statistic, the image footprint must fit in three ways. "
        "The same-row-phase segment bound requires p > 8; 16 is the smallest aligned value.",
        "srad",
        (
            "The coefficient stage covers at most 18 rows and at most 18 elements per row.",
            "The footprint proof covers the image input and one reserved statistic tag, with no simultaneous stores.",
        ),
    ),
    Preset(
        "lud", "LUD", RODINIA, "default", "512 × 512, 16 × 16 tiles",
        "Dense unpivoted LU with diagonal, perimeter, and trailing tile launches.",
        "experiment_18_lud/workload.cu and spec.py", (("matrix", 512, 512),),
        "Aligned half-line tiles",
        "The 16-float row segments already align to sectors, and adjacent half-line tiles may share a tag. "
        "Repeated factor accesses within a tile use shared memory.",
        "compact",
    ),
    Preset(
        "gaussian", "Gaussian", RODINIA, "default", "256 × 256",
        "Full Gaussian elimination and back substitution. The bound concerns the pivot factor stage.",
        "experiment_19_gaussian/workload.cu and spec.py", (("matrix", 256, 256),),
        "Pivot-column group occupancy",
        "At the first aligned increment, a cross-row factor warp uses at most two numerator tags per set. "
        "One reserved pivot broadcast still fits in four ways.",
        "first_sector",
        (
            "The bound concerns one factor warp, with a packed-pitch peak of eight numerator tags per L1 set.",
            "This is a footprint constraint, not a claim about cache misses.",
        ),
    ),
    Preset(
        "backprop", "Backprop", RODINIA, "default", "65,536 inputs, 16 hidden units",
        "One full training update. Weight and momentum matrices include the input bias row.",
        "experiment_20_backprop/workload.cu and spec.py",
        (("w1", 65537, 16), ("v1", 65537, 16)),
        "Two rows share one tag per warp",
        "An aligned full warp already reads two adjacent 16-float rows in one tag and four sectors per operand. "
        "Keeping those two rows packed minimizes that footprint.",
        "compact",
    ),
    Preset(
        "bfs", "BFS", RODINIA, "default", "65,536 vertices; degrees 3–9",
        "CSR-style adjacency storage with compact neighbors, explicit starts, and varying degree.",
        "experiment_16_bfs/workload.cu and spec.py", (),
        "No applicable rectangular pitch",
        "The adjacency rows vary in length and have no common physical pitch. "
        "The compact baseline is the only layout supported by this profile.",
        "csr",
    ),
    Preset(
        "kmeans", "K-means", RODINIA, "default", "16,384 points × 32 features, 8 centers",
        "Six Lloyd iterations with feature-major storage and a compiler-specific register reuse observation.",
        "experiment_21_kmeans/workload.cu and spec.py", (("features", 32, 16384),),
        "Point reuse is in registers",
        "In the studied sm_89 build, point values are loaded before the center loop and reused in registers. "
        "The packed global feature rows retain an aligned warp load.",
        "compact",
    ),
    Preset(
        "pathfinder", "Pathfinder", RODINIA, "default", "129 × 8192, int32",
        "A shared-memory wavefront advances eight rows per launch; only the weight-grid rows have a pitch.",
        "experiment_17_pathfinder/workload.cu and spec.py", (("wall", 129, 8192),),
        "Global weights stream once per local row",
        "Each local step reads a new row of weights while prior path costs are reused in shared memory. "
        "The known global stream already has an aligned compact pitch.",
        "compact",
    ),
)

_BY_NAME = {preset.name: preset for preset in _PROFILES}


def list_presets() -> tuple[Preset, ...]:
    """Return all supported fixed cases in study order."""
    return _PROFILES


def _srad_pad(width: int) -> int:
    """Solve the fixed 16×16 CTA halo bound with a one-line reserve."""
    tile = 16
    row_period = L1_SETS // gcd(L1_SETS, width // LINE_ELEMENTS)
    ways_available = L1_WAYS - 1
    segment = tile + 2
    # At most 'ways_available' row segments spaced by row_period*p may
    # intersect one line. The inclusive start range spans line+segment-2.
    return SECTOR_ELEMENTS * (
        (LINE_ELEMENTS + segment - 2) // (row_period * ways_available * SECTOR_ELEMENTS) + 1
    )


def analyze_preset(name: str) -> Recommendation:
    """Apply the source-specific rule for one fixed preset or raise KeyError."""
    preset = _BY_NAME[name.strip().lower()]
    source = f"{STUDY}: {preset.source}"
    scope = f"{preset.family} fixed study case. {preset.size}. {preset.summary}"
    if preset.method == "csr":
        return Recommendation(preset.operator, 0, "unsupported", preset.rule, preset.reason, scope, source=source)

    width = preset.operands[0][2]
    facts = preset.facts
    assumptions = MODULO_ASSUMPTIONS + (
        "The recommendation applies to this source implementation, fixed dimensions, and its declared window.",
        "Timing, actual cache hash functions, and other interleavings are outside the analytical rule.",
    )
    if preset.method == "adjacent":
        adjacent = analyze_adjacent_columns(width, preset.operands[0][1])
        if adjacent.p is None:
            raise ValueError(f"Preset {preset.name} does not meet the adjacent-column proof")
        pad = adjacent.p
        facts += adjacent.facts
        assumptions = adjacent.assumptions
    elif preset.method == "first_sector":
        # The source-specific signed-residue or footprint proof establishes
        # feasibility at the first aligned increment for width 256.
        if width != 256:
            raise ValueError(f"Preset {preset.name} requires its fixed 256-element width")
        pad = SECTOR_ELEMENTS
    elif preset.method == "odd_lines":
        pad = (LINE_ELEMENTS - width) % (2 * LINE_ELEMENTS)
    elif preset.method == "srad":
        pad = _srad_pad(width)
    elif preset.method == "compact":
        pad = 0
    else:
        raise ValueError(f"Unknown rule for preset {preset.name}: {preset.method}")

    layouts = tuple(RowLayout(operand, rows, cols, cols + pad) for operand, rows, cols in preset.operands)
    return Recommendation(
        preset.operator,
        pad,
        "recommend_padding" if pad else "keep_baseline",
        preset.rule,
        preset.reason,
        scope,
        layouts,
        assumptions,
        facts,
        source,
    )
