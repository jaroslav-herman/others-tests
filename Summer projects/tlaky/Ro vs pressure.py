import wepy.basics as we
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from datetime import datetime
from pathlib import Path

file = r"C:\Users\Herman\OneDrive - Univerzita Karlova\Letní projekt\Tlaky\432.csv"
measurement_folder = Path(
r'\\ELECTROLYZER\PEM-WE_measurements\2026\432_III_III_IrOxonPTL_150nm_Pt_500ug_Pressures_Pressureincrease-0-5bar'
)

data = we.read_file(file, delimiter=',',skiprows=0)

# Remove a possible UTF-8 BOM from the source-data column name.
data.columns = [str(column).lstrip("\ufeff").replace("ï»¿", "") for column in data.columns]
if "source_file" not in data.columns:
    raise KeyError("The data table does not contain a 'source_file' column.")

pressure_pattern = re.compile(
    r"_(?P<first>[+-]?\d+(?:[.,]\d+)?)alebo"
    r"(?P<second>[+-]?\d+(?:[.,]\d+)?)_bar",
    re.IGNORECASE,
)


def extract_numbers(filename: str) -> tuple[float, float]:
    """Extract the two pressure values exactly as in the sorting script."""
    match = pressure_pattern.search(str(filename))
    if match is None:
        return np.nan, np.nan

    return (
        float(match.group("first").replace(",", ".")),
        float(match.group("second").replace(",", ".")),
    )


# Extract the two pressure values directly from each source filename.
extracted_pressure = pd.DataFrame(
    data["source_file"].map(extract_numbers).tolist(),
    index=data.index,
    columns=["pressure_before_alebo", "pressure_after_alebo"],
)
data[["pressure_before_alebo", "pressure_after_alebo"]] = extracted_pressure


def basename(value: object) -> str:
    """Return a filename basename for either slash convention."""
    return str(value).strip().strip('"\'').replace("\\", "/").rsplit(
        "/", maxsplit=1
    )[-1]


mps_files = list(measurement_folder.glob("*.mps"))


def modified_date_for_source(source_file: object) -> pd.Timestamp:
    """Find the matching .mps file and return its filesystem modified date."""
    source_name = basename(source_file).casefold()
    matches = [
        path
        for path in mps_files
        if source_name == path.name.casefold()
        or source_name.startswith(f"{path.stem.casefold()}_")
    ]
    if not matches:
        return pd.NaT

    # Prefer the most specific matching .mps name if several are present.
    match = max(matches, key=lambda path: len(path.stem))
    return pd.Timestamp(datetime.fromtimestamp(match.stat().st_mtime))


data["modified_date"] = data["source_file"].map(modified_date_for_source)
data.sort_values("modified_date", inplace=True, na_position="last")
data.reset_index(drop=True, inplace=True)


groups = data.groupby('source_file', sort=False)

R0s = []
pressure = []
for file,group in groups:
    valid_pressure = group['pressure_after_alebo'].dropna()
    if valid_pressure.empty:
        continue

    R0s.append(np.mean(group['R0']))
    pressure.append(np.mean(valid_pressure))


plt.plot(pressure,R0s)
plt.yscale('log')

# Plot the average R0 for each source file against its .mps modified date.
average_r0_by_date = (
    data.dropna(subset=["modified_date"])
    .groupby(["source_file", "modified_date"], as_index=False)["R0"]
    .mean()
    .sort_values("modified_date")
)

plt.figure()
plt.plot(
    average_r0_by_date["modified_date"],
    average_r0_by_date["R0"],
    marker="o",
)
plt.xlabel("Modified date")
plt.ylabel("Average R0")
plt.title("Average R0 versus modified date")
plt.yscale('log')
plt.gcf().autofmt_xdate()
plt.show()
