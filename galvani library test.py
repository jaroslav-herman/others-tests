from galvani import BioLogic
import pandas as pd
import wepy.basics as we
file = r"\\ELECTROLYZER\PEM-WE_measurements\2026\439_III_III_IrOxonPTL_150nm_Pt_500ug_Pressures_N115_etchedcathode_BDC929\III_Day1_Procedure1_05_PEIS_C01.mpr"
file = r"\\ELECTROLYZER\PEM-WE_measurements\2026\430_VIII_VIII_IrOx(6,15)_60min50WTiCoverlayer,refel_N115_etchedcathode\VIII_Day1_Procedure1_05_PEIS_C01.mpr"
mpr_file = BioLogic.MPRfile(file)
df = pd.DataFrame(mpr_file.data)
print(df.columns)

df_mpt = we.read_file(r"\\ELECTROLYZER\PEM-WE_measurements\2026\430_VIII_VIII_IrOx(6,15)_60min50WTiCoverlayer,refel_N115_etchedcathode\VIII_Day1_Procedure1_05_PEIS_C01.mpt")
print(df_mpt.columns)