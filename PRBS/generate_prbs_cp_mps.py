"""Generate a two-band PRBS chronopotentiometry procedure for EC-Lab.

The procedure is deliberately represented as explicit CP steps, matching the
format used by the supplied EC-Lab setting file. It applies a deterministic
maximal-length PRBS around 500 mA at two segment durations while recording
instantaneous Ewe. Start with 1 ms recording and validate the actual timestamps:
the previous 0.2 ms files contained extensive gaps. The fast band uses PRBS-10
with 4 ms segments. Time controls step duration; the charge cutoff is disabled.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from scipy.signal import max_len_seq


DEFAULT_TEMPLATE = (
    Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova")
    / "VIII_Day9_CP_500mA.mps"
)
DEFAULT_OUTPUT = Path("PRBS_500mA_two_band_CP_verified.mps")


def maximal_length_prbs(order: int) -> list[int]:
    """Return one deterministic maximal-length binary sequence.

    Use SciPy's primitive polynomials. The former two-tap recurrence for order
    nine repeated after just 73 bits; checking only that both bits occur did
    not detect that error.
    """
    if order not in (8, 9, 10, 11, 14, 15, 16):
        raise ValueError("Supported PRBS orders: 8, 9, 10, 11, 14, 15, 16")
    sequence = max_len_seq(order)[0].tolist()
    # Every nonzero length-n state must occur exactly once, including wrap.
    extended = sequence + sequence[:order-1]
    states = {tuple(extended[j:j+order]) for j in range(len(sequence))}
    if len(states) != 2**order-1 or sum(sequence) != 2**(order-1):
        raise RuntimeError("Generated sequence is not maximal length")
    return sequence


def dqm_mah(current_ma: float, duration_s: float) -> float:
    """Charge passed during one step in mA h, matching EC-Lab units."""
    return current_ma * duration_s / 3600.0


def compress_runs(sequence: list[int], segment_s: float) -> list[tuple[int, float]]:
    """Combine adjacent equal PRBS bits into one longer CP step."""
    runs: list[tuple[int, float]] = []
    for bit in sequence:
        if runs and runs[-1][0] == bit:
            runs[-1] = (bit, runs[-1][1] + segment_s)
        else:
            runs.append((bit, segment_s))
    return runs


def technique_from_runs(
    number: int,
    runs: list[tuple[int | None, float]],
    center_ma: float,
    amplitude_ma: float,
    goto: list[int] | None = None,
    cycles: list[int] | None = None,
    record_interval_s: float = .001,
) -> str:
    goto = goto or [0] * len(runs)
    cycles = cycles or [0] * len(runs)
    if len(goto) != len(runs) or len(cycles) != len(runs):
        raise ValueError("goto and cycles must have one value per CP step")
    currents = [
        center_ma if bit is None else center_ma + amplitude_ma if bit else center_ma - amplitude_ma
        for bit, _ in runs
    ]
    durations = [duration for _, duration in runs]
    # EC-Lab stores each step in a fixed 20-character column.  Using a fixed
    # field width is essential: joining values with a fixed number of spaces
    # shifts columns whenever a value has more than one character.
    field = lambda values: "".join(f"{str(value):<20}" for value in values)
    return (
        f"Technique : {number}\n"
        "Chronopotentiometry\n"
        "Ns                  " + field(range(len(runs))) + "\n"
        "Is                  " + field(f"{value:.3f}" for value in currents) + "\n"
        "unit Is             " + field("mA" for _ in runs) + "\n"
        "vs.                 " + field("<None>" for _ in runs) + "\n"
        "ts (h:m:s)          " + field(f"0:00:{duration:0.4f}" for duration in durations) + "\n"
        "EM (V)              " + field("2.000" for _ in runs) + "\n"
        "dQM                 " + field("0.000000" for _ in runs) + "\n"
        "unit dQM            " + field("mA.h" for _ in runs) + "\n"
        "record              " + field("Ewe" for _ in runs) + "\n"
        "dEs (mV)            " + field("0.00" for _ in runs) + "\n"
        "dts (s)             " + field(f"{record_interval_s:.4f}" for _ in runs) + "\n"
        "E range min (V)     " + field("-10.000" for _ in runs) + "\n"
        "E range max (V)     " + field("10.000" for _ in runs) + "\n"
        "I Range             " + field("1 A" for _ in runs) + "\n"
        "Bandwidth           " + field("4" for _ in runs) + "\n"
        "goto Ns'            " + field(goto) + "\n"
        "nc cycles           " + field(cycles) + "\n"
    )


def generate(
    template: Path,
    output: Path,
    center_ma: float = 500.0,
    amplitude_ma: float = 10.0,
    settle_s: float = 30.0,
    slow_order: int = 9,
    slow_segment_s: float = 0.010,
    slow_cycles: int = 12,
    fast_order: int = 10,
    fast_segment_s: float = 0.004,
    fast_cycles: int = 7,
    record_interval_s: float = .001,
) -> None:
    if record_interval_s < .0002 or abs(record_interval_s/.0001-round(record_interval_s/.0001)) > 1e-6:
        raise ValueError("Recording interval must be >=0.2 ms on the 0.1 ms settings grid")
    if min(slow_segment_s, fast_segment_s) < 4 * record_interval_s:
        raise ValueError("Use at least four recorded samples per shortest PRBS bit")
    if center_ma-amplitude_ma <= 0 or center_ma+amplitude_ma > 1000:
        raise ValueError("The fixed 1 A range requires 0 < current <= 1000 mA")
    # EC-Lab files are commonly ANSI/Windows-1252; latin-1 preserves every byte
    # in the supplied header, including legacy degree/micro symbols.
    source = template.read_text(encoding="latin-1")
    header = source.split("Technique : 1", 1)[0]
    if "Potential control : Ewe\n" not in header:
        raise ValueError("Use an Ewe-control template, as requested for these measurements")
    # Keep the source header unchanged.  Both PRBS bands are placed in one CP
    # technique, so the source file's OCV-between-techniques setting is unused.

    # Store one explicit maximal-length period and let EC-Lab repeat it using
    # the final step's goto/cycle fields. This avoids enormous files while
    # preserving the complete fast sequence instead of repeating PRBS-9.
    slow = compress_runs(maximal_length_prbs(slow_order), slow_segment_s)
    fast = compress_runs(maximal_length_prbs(fast_order), fast_segment_s)
    settle = [(None, settle_s)]
    runs = settle + slow + fast
    slow_start = 1
    slow_end = slow_start + len(slow) - 1
    fast_start = slow_end + 1
    fast_end = fast_start + len(fast) - 1
    goto = [0] * len(runs)
    cycles = [0] * len(runs)
    # The final step of each block loops back to the first step in that block.
    # The first pass is followed by the requested number of repetitions.
    goto[slow_end] = slow_start
    cycles[slow_end] = slow_cycles
    goto[fast_end] = fast_start
    cycles[fast_end] = fast_cycles
    content = header + technique_from_runs(1, runs, center_ma, amplitude_ma, goto, cycles, record_interval_s)
    # EC-Lab setting files use Windows CRLF line endings.
    output.write_bytes(content.replace("\n", "\r\n").encode("latin-1"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--center-ma", type=float, default=500.0)
    parser.add_argument("--amplitude-ma", type=float, default=10.0)
    parser.add_argument("--settle-s", type=float, default=30.0)
    parser.add_argument("--slow-order", type=int, default=9)
    parser.add_argument("--slow-segment-ms", type=float, default=10.0)
    parser.add_argument("--slow-cycles", type=int, default=12)
    parser.add_argument("--fast-order", type=int, default=10)
    parser.add_argument("--fast-segment-ms", type=float, default=4.0)
    parser.add_argument("--fast-cycles", type=int, default=7)
    parser.add_argument("--record-ms", type=float, default=1.0,
                        help="Recording interval; start at 1 ms and verify actual timestamps")
    args = parser.parse_args()
    generate(
        args.template,
        args.output,
        args.center_ma,
        args.amplitude_ma,
        args.settle_s,
        args.slow_order,
        args.slow_segment_ms / 1000.0,
        args.slow_cycles,
        args.fast_order,
        args.fast_segment_ms / 1000.0,
        args.fast_cycles,
        args.record_ms / 1000.,
    )
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
