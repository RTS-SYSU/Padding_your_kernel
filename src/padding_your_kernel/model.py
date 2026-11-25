"""Deterministic padding rules for bounded row-major cache reuse models."""

from dataclasses import dataclass
from typing import Literal


LINE_ELEMENTS = 32
SECTOR_ELEMENTS = 8
WARP_LANES = 32
L1_SETS = 32
L1_WAYS = 4
L2_PARTITION_SETS = 64
MAX_DIMENSION = 2**31 - 1

Status = Literal["recommend_padding", "keep_baseline", "unsupported"]

MODULO_ASSUMPTIONS = (
    "The ideal L1 has 32 sets, 4 ways and 128-byte lines with direct modulo indexing and LRU replacement.",
    "The ideal L2 uses a single 64-set, 16-way partition with direct modulo indexing. "
    "Device slice mapping and the stream filtered by L1 are outside this model.",
)


@dataclass(frozen=True, slots=True)
class RowLayout:
    """Physical pitch and additional space for one row-major operand."""

    operand: str
    rows: int
    logical_width: int
    physical_pitch: int
    element_bytes: int = 4

    @property
    def extra_bytes(self) -> int:
        return self.rows * (self.physical_pitch - self.logical_width) * self.element_bytes

    def to_dict(self) -> dict:
        return {
            "operand": self.operand,
            "rows": self.rows,
            "logical_width": self.logical_width,
            "physical_pitch": self.physical_pitch,
            "element_bytes": self.element_bytes,
            "extra_bytes": self.extra_bytes,
        }


@dataclass(frozen=True, slots=True)
class Recommendation:
    """A single layout decision within the stated analytical scope."""

    operator: str
    p: int | None
    status: Status
    rule: str
    reason: str
    scope: str
    layouts: tuple[RowLayout, ...] = ()
    assumptions: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    source: str | None = None

    @property
    def extra_bytes(self) -> int | None:
        return None if self.p is None else sum(layout.extra_bytes for layout in self.layouts)

    def to_dict(self) -> dict:
        return {
            "operator": self.operator,
            "p": self.p,
            "unit": "elements_per_row",
            "status": self.status,
            "rule": self.rule,
            "reason": self.reason,
            "scope": self.scope,
            "layouts": [layout.to_dict() for layout in self.layouts],
            "assumptions": list(self.assumptions),
            "facts": list(self.facts),
            "source": self.source,
            "extra_bytes": self.extra_bytes,
        }


def adjacent_matrix_reuse_distance(lane: int, phase: int, pad: int) -> int:
    """Different same-set matrix lines between same-line adjacent-column accesses.

    The row width must be a multiple of 2048 floats in the fixed cache
    geometries. Phase is the target element's float offset within its cache line,
    and lane is its index in an ascending row warp.
    """
    if type(lane) is not int or not 0 <= lane < WARP_LANES:
        raise ValueError("lane must be an integer from 0 through 31")
    if type(phase) is not int or not 0 <= phase < LINE_ELEMENTS - 1:
        raise ValueError("phase must be an integer from 0 through 30 for same-line reuse")
    if type(pad) is not int or not 0 <= pad <= LINE_ELEMENTS or pad % SECTOR_ELEMENTS:
        raise ValueError("pad must be an integer multiple of 8 from 0 through 32")
    if pad == 0:
        return WARP_LANES - 1
    return (
        min(WARP_LANES - 1 - lane, (LINE_ELEMENTS - 1 - phase) // pad)
        + min(lane, (phase + 1) // pad)
    )


def _peak_adjacent_distance(pad: int) -> int:
    """Exact worst distance for the supported aligned pitches, without a scan."""
    if pad == 0:
        return WARP_LANES - 1
    if pad == LINE_ELEMENTS:
        return 0  # Both numerators in the reuse equation are less than one line.
    return LINE_ELEMENTS // pad


def analyze_adjacent_columns(
    width: int,
    rows: int,
    *,
    reserved_lines: int = 1,
    base_alignment_bytes: int = 128,
) -> Recommendation:
    """Solve one pitch for a regular FP32 adjacent-column row-warp trace."""
    for name, value, minimum in (
        ("width", width, 1),
        ("rows", rows, 1),
        ("reserved_lines", reserved_lines, 0),
        ("base_alignment_bytes", base_alignment_bytes, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if max(width, rows) > MAX_DIMENSION:
        raise ValueError(f"width and rows must be at most {MAX_DIMENSION}")

    scope = "One ascending 32-lane row warp reusing the same matrix tag at consecutive columns."
    assumptions = MODULO_ASSUMPTIONS + (
        "Use a row-major FP32 matrix with consecutive rows assigned to lanes and a fixed ordered trace. "
        "A partial final warp cannot add intervening lines.",
        f"At most {reserved_lines} other distinct lines in any target set may intervene between two matrix accesses.",
        "The proof covers same-tag reuse. Cross-warp accesses, sector validity and timing require separate evidence.",
    )
    unsupported = None
    if width % (LINE_ELEMENTS * L2_PARTITION_SETS):
        unsupported = "Logical width must be a multiple of 2048 FP32 elements for the row-stride term to cancel at both levels."
    elif rows < WARP_LANES:
        unsupported = "The custom rule requires at least 32 rows so its full-warp certificate applies."
    elif base_alignment_bytes % (4 * SECTOR_ELEMENTS):
        unsupported = "The stated base alignment must guarantee a 32-byte aligned row start."
    elif reserved_lines >= L1_WAYS:
        unsupported = "The reserved competing lines occupy every L1 way in the worst case."
    if unsupported is not None:
        return Recommendation(
            "Custom row-strided reuse", None, "unsupported", "Adjacent-column reuse",
            unsupported, scope, assumptions=assumptions,
        )

    available = L1_WAYS - reserved_lines
    # floor(32 / p) < available. At p=32, the exact maximum is zero,
    # so this injective stride also covers the last remaining way.
    strict_bound = SECTOR_ELEMENTS * (LINE_ELEMENTS // (SECTOR_ELEMENTS * available) + 1)
    pad = min(LINE_ELEMENTS, strict_bound)
    peak = _peak_adjacent_distance(pad)
    previous = _peak_adjacent_distance(pad - SECTOR_ELEMENTS)
    return Recommendation(
        "Custom row-strided reuse",
        pad,
        "recommend_padding",
        "Adjacent-column reuse",
        f"The smallest 32-byte-aligned row increment is {pad} floats. "
        f"It leaves at most {peak} same-set matrix interveners plus {reserved_lines} reserved lines, "
        f"within the {L1_WAYS}-way L1 reuse limit.",
        scope,
        (RowLayout("matrix", rows, width, width + pad),),
        assumptions,
        (
            "The unpadded warp has 31 same-set matrix interveners on a within-line reuse.",
            f"The preceding aligned pad has a worst-case matrix reuse distance of {previous}.",
            f"The selected layout has a worst-case same-set distance of {peak + reserved_lines} "
            f"with the stated reserve. LRU requires distance < {L1_WAYS}.",
        ),
    )
