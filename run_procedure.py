"""Run all generated FTACV settings through EC-Lab OLE-COM.

Every ``.mps`` file is loaded in a random order. The measurement result is
written beside its settings file, for example ``FTACV.mps`` -> ``FTACV.mpr``.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS_ROOT = Path(
    r"C:\Users\Herman\OneDrive - Univerzita Karlova\FTACV\parameter range"
)
EC_LAB_VERSION = "11.71"


def find_settings(root: Path) -> list[Path]:
    """Find all generated settings files below *root*."""
    return sorted(path for path in root.rglob("*.mps") if path.is_file())


def build_channel(device_id: int, channel_id: int) -> Any:
    from biocom.com.server import DeviceChannel
    from biocom.mps.common import BLDeviceModel

    return DeviceChannel(
        device_id,
        channel_id,
        BLDeviceModel.SP150e,
        name=f"SP150_Channel{channel_id}",
    )


def run_measurements(
    settings_root: Path,
    device_id: int,
    channel_id: int,
    seed: int | None,
    timeout: float,
    interval: float,
) -> int:
    settings_files = find_settings(settings_root)
    if not settings_files:
        raise FileNotFoundError(f"No .mps files found below {settings_root}")

    rng = random.Random(seed)
    rng.shuffle(settings_files)

    print(f"Found {len(settings_files)} settings files")
    print(f"Random seed: {seed if seed is not None else 'system-generated'}")

    from biocom.com.server import OLECOM
    from biocom.mps.config import set_versions

    set_versions(EC_LAB_VERSION)
    server = OLECOM()
    server.launch_server()
    channel = build_channel(device_id, channel_id)
    print(f"Using {channel}")

    failures: list[tuple[Path, Exception]] = []
    for number, mps_file in enumerate(settings_files, start=1):
        mpr_file = mps_file.with_suffix(".mpr")
        print(f"[{number}/{len(settings_files)}] {mps_file}")
        print(f"             -> {mpr_file}")

        try:
            server.load_settings(channel, str(mps_file))
            server.run_channel(channel, str(mpr_file))
            result = server.wait_for_channel(
                channel,
                min_wait=10.0,
                timeout=timeout,
                interval=interval,
            )
            print(f"             finished: {result.name}")
        except Exception as error:  # keep the batch moving and report failures at the end
            failures.append((mps_file, error))
            print(f"             FAILED: {error}", file=sys.stderr)

    if failures:
        print(f"Completed with {len(failures)} failure(s):", file=sys.stderr)
        for mps_file, error in failures:
            print(f"  {mps_file}: {error}", file=sys.stderr)
        return 1

    print("All measurements completed successfully")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "settings_root",
        type=Path,
        nargs="?",
        default=DEFAULT_SETTINGS_ROOT,
        help="Folder containing the generated .mps files",
    )
    parser.add_argument("--device-id", type=int, default=3)
    parser.add_argument("--channel-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible random-order seed")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Timeout per measurement in seconds")
    parser.add_argument("--interval", type=float, default=5.0, help="Status polling interval in seconds")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the randomized order without connecting to EC-Lab",
    )
    args = parser.parse_args()

    settings_files = find_settings(args.settings_root)
    if not settings_files:
        parser.error(f"No .mps files found below {args.settings_root}")

    if args.dry_run:
        rng = random.Random(args.seed)
        rng.shuffle(settings_files)
        for number, path in enumerate(settings_files, start=1):
            print(f"{number:03d}: {path} -> {path.with_suffix('.mpr')}")
        return 0

    return run_measurements(
        args.settings_root,
        args.device_id,
        args.channel_id,
        args.seed,
        args.timeout,
        args.interval,
    )


if __name__ == "__main__":
    raise SystemExit(main())
