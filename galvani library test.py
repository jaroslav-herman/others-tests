# %%

import re
import difflib

from galvani import BioLogic
import numpy as np
import pandas as pd
import wepy.basics as we


def read_mpr_cycle_number(path):
    """Read the MPR cycle field even when Galvani sees newer column IDs.

    Galvani identifies cycle number as column ID 24.  Recent EC-Lab files can
    contain additional IDs that make ``BioLogic.MPRfile`` fail before it
    returns any data.  The cycle field occurs before those newer fields, so it
    can be extracted safely from the record using the IDs known to Galvani.
    """
    with open(path, "rb") as mpr_handle:
        mpr_handle.read(len(BioLogic.MPR_MAGIC))
        modules = list(BioLogic.read_VMP_modules(mpr_handle))

    data_module = next(m for m in modules if m["shortname"] == b"VMP data  ")
    n_points = int.from_bytes(data_module["data"][:4], "little")
    n_columns = data_module["data"][4]

    if data_module["version"] != 0:
        raise ValueError("Fallback currently supports MPR data-module version 0 only")

    raw_ids = data_module["data"][5 : 5 + 2 * n_columns]
    column_ids = raw_ids[1::2] if raw_ids[0] == 0 else raw_ids[:n_columns]
    cycle_position = None
    record_size = 0

    # Build the byte layout up to column ID 24.  All fields before it are
    # supported by Galvani, so their sizes are unambiguous.
    for column_id in column_ids:
        if column_id == 24:
            cycle_position = record_size
            break
        if column_id in BioLogic.VMPdata_colID_dtype_map:
            _, dtype = BioLogic.VMPdata_colID_dtype_map[int(column_id)]
            record_size += np.dtype(dtype).itemsize
        else:
            raise ValueError(f"Unknown column ID {column_id} before cycle number")

    if cycle_position is None:
        raise ValueError("MPR file has no cycle-number column (column ID 24)")

    # The total record size is recoverable from the data-module payload.
    record_data = data_module["data"][1007:]
    record_size = len(record_data) // n_points
    return np.ndarray(
        n_points,
        dtype="<f8",
        buffer=record_data,
        offset=cycle_position,
        strides=(record_size,),
    )


def normalize_column_name(name):
    """Turn BioLogic names into a comparable key despite formatting drift."""
    text = str(name).strip().lower()

    # Remove formatting noise that does not change the physical quantity.
    text = text.replace("<", "").replace(">", "")
    text = text.replace("|", "")
    text = text.replace(" ", "")
    text = text.replace("-im", "im")
    text = text.replace("-re", "re")
    text = text.replace("-", "")

    # Normalize common unit / symbol variants.
    text = text.replace("ohm", "ohm")
    text = text.replace("µ", "u")
    text = text.replace("μ", "u")
    text = text.replace("/", "")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("[", "").replace("]", "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace("%", "pct")
    text = text.replace("deg", "deg")

    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def match_columns(df_mpr, df_mpt):
    """Return a mapping with the same physical quantity on both dataframes."""
    mpr_norm = {
        normalize_column_name(col): col
        for col in df_mpr.columns
        if str(col).strip() != ""
    }
    mpt_norm = {
        normalize_column_name(col): col
        for col in df_mpt.columns
        if str(col).strip() != ""
    }

    matched = {}
    for mpt_col in df_mpt.columns:
        key = normalize_column_name(mpt_col)
        if key in mpr_norm:
            matched[mpt_col] = mpr_norm[key]
            continue

        # Fuzzy fallback for small notational differences that still represent the same quantity.
        best_match = None
        best_score = 0.0
        for mpr_key, mpr_col in mpr_norm.items():
            score = difflib.SequenceMatcher(None, key, mpr_key).ratio()
            if score > best_score:
                best_score = score
                best_match = mpr_col

        if best_score >= 0.85:
            matched[mpt_col] = best_match

    return matched


# file = r"\\ELECTROLYZER\PEM-WE_measurements\2026\439_III_III_IrOxonPTL_150nm_Pt_500ug_Pressures_N115_etchedcathode_BDC929\III_Day1_Procedure1_05_PEIS_C01.mpr"
file = r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\Data z PEMWE\VIII_Day3_Procedure1_05_PEIS_C01.mpr"

# Galvani can fail on some newer BioLogic files, so keep the script usable even when MPR parsing breaks.
try:
    mpr_file = BioLogic.MPRfile(file)
    df_mpr = pd.DataFrame(mpr_file.data)
except Exception as exc:  # pragma: no cover - diagnostic script
    print(f"MPR load failed: {exc}")
    df_mpr = pd.DataFrame({"cycle number": read_mpr_cycle_number(file)})

# MPT is the reliable source for this dataset in this project.
df_mpt = we.read_file(
    r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\Data z PEMWE\VIII_Day3_Procedure1_05_PEIS_C01.mpt"
)

matched = match_columns(df_mpr, df_mpt)
mpr_to_mpt = {mpr_col: mpt_col for mpt_col, mpr_col in matched.items()}

print("Matched columns:")
if mpr_to_mpt:
    for mpr_col, mpt_col in mpr_to_mpt.items():
        print(f"  {mpr_col!r} -> {mpt_col!r}")
else:
    print("  No MPR columns were matched because the MPR parser failed for this file.")

print("\nUnmatched MPT columns:")
for col in df_mpt.columns:
    if col not in matched:
        print(f"  {col!r}")

# Rename MPT columns to the MPR-equivalent names when the same quantity is found.
df_mpt_aligned = df_mpt.rename(
    columns={mpt_col: mpr_col for mpt_col, mpr_col in matched.items()}
)
print("\nAligned MPT columns sample:")
print(list(df_mpt_aligned.columns[:20]))

# %%

file = r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\Data z PEMWE\VIII_Day3_Procedure1_05_PEIS_C01.mpr"
if "cycle number" not in df_mpr:
    df_mpr["cycle number"] = read_mpr_cycle_number(file)
print(df_mpr.columns)

for cycle in df_mpr['cycle number'].unique():
    df_cycle = df_mpr[df_mpr['cycle number'] == cycle]
    print(f"Cycle {cycle}: {len(df_cycle)} rows")
# %%
print(df_mpr['cycle number'])
# %%
for column in df_mpr.columns:
    print(f"{column!r}: {len(df_mpr[column].unique())}")
# %%
df_mpt = we.read_file(
    r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\Data z PEMWE\VIII_Day3_Procedure1_05_PEIS_C01.mpt"
)
for column in df_mpr.columns:
    print(f"{column!r}: {len(df_mpt[column].unique())}")

# %%
