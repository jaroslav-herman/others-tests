import wepy.basics as we

file = r"\\ELECTROLYZER\PEM-WE_measurements\2026\467_III_cathode_etching_series_20min\VIII_Day10_Procedure1_05_PEIS_C01.mpr"
# MPR files are binary Bio-Logic files.  Some newer files contain columns that
# the installed Galvani version does not know yet, so keep the known columns.
data = we.read_mpr(file)
print(data.columns)
