# %%
import numpy as np
import pandas as pd
from galvani import BioLogic
import wepy.basics as we
import matplotlib.pyplot as plt
#from IPython import get_ipython
#ip = get_ipython()

#if ip is not None:
#    ip.run_line_magic('load_ext', 'autoreload')
#    ip.run_line_magic('autoreload', '2')
# %%

# %%
'''
MPR_MAGIC = b"BIO-LOGIC MODULAR FILE\x1a".ljust(48) + b"\x00\x00\x00\x00"


def read_cycle_number_from_mpr(path):
    """Read column ID 24 directly from an MPR data module.

    This avoids decoding the complete record, which is important for newer
    EC-Lab files containing column IDs that this Galvani version does not know.
    In this MPR format, column ID 24 is an eight-byte little-endian float.
    """
    with open(path, "rb") as handle:
        if handle.read(len(MPR_MAGIC)) != MPR_MAGIC:
            raise ValueError("Not a Bio-Logic MPR file")
        data_module = next(
            module
            for module in BioLogic.read_VMP_modules(handle)
            if module["shortname"] == b"VMP data  "
        )
        module_data = data_module["data"]
        data_version = int(data_module["version"])

    n_points = int(np.frombuffer(module_data[:4], dtype="<u4", count=1)[0])
    n_columns = int(module_data[4])
    if data_version != 0:
        raise ValueError("This direct reader expects a version-0 data module")

    # Version-0 files store each column ID in a two-byte slot: ID, zero.
    column_ids = np.frombuffer(module_data[5 : 5 + 2 * n_columns], dtype="u1")[1::2]
    try:
        cycle_index = list(column_ids).index(24)
    except ValueError as exc:
        raise ValueError("MPR file has no cycle-number column (ID 24)") from exc

    # The fields before ID 24 have fixed sizes in this file.
    sizes = {4: 8, 24: 8, 39: 2, 131: 2, 13: 8, 215: 8}
    cycle_offset = sum(
        sizes.get(int(column_id), 4) for column_id in column_ids[:cycle_index]
    )

    record_data = module_data[1007:]
    record_size, remainder = divmod(len(record_data), n_points)
    if remainder:
        raise ValueError("MPR data records have an invalid size")

    return np.ndarray(
        shape=n_points,
        dtype="<f8",
        buffer=record_data,
        offset=cycle_offset,
        strides=(record_size,),
    )
'''
# %%


file_mpr = r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\Data z PEMWE\VIII_Day3_Procedure1_05_PEIS_C01.mpr"
#file_mpr = r"C:\Users\Herman\OneDrive - Univerzita Karlova\Ti overlayer\159_II_II_Ti400nm + IrOx Ar14 O1, etched, Ir 25nm, RDC789\II_Day13_procedure1_06_PEIS_C02.mpr"
file_mpt = r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\Data z PEMWE\VIII_Day3_Procedure1_05_PEIS_C01.mpt"

df_mpr = pd.DataFrame(BioLogic.MPRfile(file_mpr).data)
df_mpt = we.read_file(file_mpt)

print(df_mpr["cycle number"])


# %%
colors = we.get_colors(len(df_mpr["cycle number"].unique()))
#for cycle, color in zip(df_mpr["cycle number"].unique(), colors):
#    df_cycle_number = df_mpr[df_mpr["cycle number"] == cycle]
#    plt.plot(
#        df_cycle_number["Re(Zwe-ce)/Ohm"], df_cycle_number["-Im(Zwe-ce)/Ohm"], c=color
#    )
#plt.gca().set_aspect("equal")
#plt.show()
# %%
plt.plot(df_mpr["cycle number"],df_mpr["<I>/mA"])

plt.show()

plt.plot(df_mpr["<I>/mA"],df_mpr["<Ewe>/V"]- df_mpr["<Ece>/V"]-df_mpt['<Ewe-Ece>/V'])
#plt.plot(df_mpr["<I>/mA"],)
plt.show()
plt.plot(df_mpr["time/s"],df_mpr["<I>/mA"])

plt.show()
plt.plot(df_mpr["cycle number"],df_mpr["I Range"])

# %%
for column in df_mpr.columns:
    print(column)