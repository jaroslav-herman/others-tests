import scipy as sc
import pylab as py
import numpy as np
import pandas as pd
import os
# import xrayutilities as xu 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob # The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order
#import sys
import re
import copy
import collections.abc
import contextlib
import warnings
from scipy.stats import linregress
import csv
from brokenaxes import brokenaxes
from impedance.models.circuits import CustomCircuit



def loadFile (path):
    import numpy as np
    return np.genfromtxt(path,comments='#')

def saveDat(data,filein):
    import numpy as np
    split=filein.split('.')
    print(data)
    np.savetxt(split[0]+'_TT.'+split[1],data,fmt=['%3.5f','%3.5f','%3.5f'])

# fmt bounds the format of the output file. Old notation %3.5f means three spaces from the left
# so the numbers e.g. 3, 30, 300 has the same alignment. 5 is the number of the decimal places.

def loadFiles(folder,wild): #folder that should be searched. wild is the constraint for the wild card (* all entries, ? only one)
    data=[] # loaded data
    filelist=[] # list of the datas in data, names with the complete paths
    filename=[] # list of the names only in the folder
    os.chdir(folder)
    qfilelist = glob.glob(wild)
    for i,ffile in enumerate(qfilelist):
        qfilelist[i] = os.path.join(folder,ffile)
    qfilelist=sorted(qfilelist)
    for n in qfilelist:
        if n[-4:] == '.dat':
            filelist.append(str(n))
            filename.append(os.path.basename(str(n)))
            data.append(np.genfromtxt(str(n),comments='#'))
    return data,filelist,filename

def loadFilesTXT(folder,wild, skiph): #folder that should be searched. wild is the constraint for the wild card (* all entries, ? only one)
    data=[] # loaded data
    labels={}
    filelist=[] # list of the datas in data, names with the complete paths
    filename=[] # list of the names only in the folder
    os.chdir(folder)
    qfilelist = glob.glob(wild)
    for i,ffile in enumerate(qfilelist):
        qfilelist[i] = os.path.join(folder,ffile)
    qfilelist=sorted(qfilelist)
    for n in qfilelist:
        if n[-4:] == '.txt':
            filelist.append(str(n))
            filename.append(os.path.basename(str(n)))
            label=(pd.read_csv(str(n),sep='\t', on_bad_lines='skip', nrows=skiph-1))
            nname=label
            print(nname)
            labels.update({os.path.basename(str(n)) : nname})
            data.append(np.genfromtxt(str(n),comments='#', delimiter='\t', skip_header=skiph))
    return data,filelist,filename,labels

def loadFilesGEN(folder,wild, skiph, form): #folder that should be searched. wild is the constraint for the wild card (* all entries, ? only one)
    data=[] # loaded data
    labels={}
    filelist=[] # list of the datas in data, names with the complete paths
    filename=[] # list of the names only in the folder
    os.chdir(folder)
    qfilelist = glob.glob(wild)
    for i,ffile in enumerate(qfilelist):
        qfilelist[i] = os.path.join(folder,ffile)
    qfilelist=sorted(qfilelist)
    for n in qfilelist:
        if n[-4:] == form:
            filelist.append(str(n))
            filename.append(os.path.basename(str(n)))
            label=(pd.read_csv(str(n),sep='\t', on_bad_lines='skip', nrows=skiph))
            nname=label
            print(nname)
            labels.update({os.path.basename(str(n)) : nname})
            data.append(np.genfromtxt(str(n),comments='#', delimiter='\t', skip_header=skiph))
    return data,filelist,filename,labels

def loadFilesDATXPS(folder,wild): #folder that should be searched. wild is the constraint for the wild card (* all entries, ? only one)
    data=[] # loaded data
    labels={}
    filelist=[] # list of the datas in data, names with the complete paths
    filename=[] # list of the names only in the folder
    os.chdir(folder)
    qfilelist = glob.glob(wild)
    for i,ffile in enumerate(qfilelist):
        qfilelist[i] = os.path.join(folder,ffile)
    qfilelist=sorted(qfilelist)
    for n in qfilelist:
        if n[-4:] == '.dat':
            filelist.append(str(n))
            filename.append(os.path.basename(str(n)))
            label=(pd.read_csv(str(n),sep='\t', engine='python', encoding="ISO8859"))
            nname=label.columns
            print(nname)
            labels.update({os.path.basename(str(n)) : nname})
            data.append(np.genfromtxt(str(n),comments='#', delimiter='\t', skip_header=1))
    return data,filelist,filename,labels

def dayproc(filelist):
    import re

    day_procedures = {}

    for filef in filelist:
        # first number = day
        day_match = re.search(r'\d+', filef)

        # procedure with optional channel (_C01)
        procedure_match = re.search(
            r'(\w\w)(?:_C\d+)?\.mpt$', filef
        )

        if day_match and procedure_match:
            day = int(day_match.group())
            procedure = procedure_match.group(1)

            day_procedures.setdefault(day, [])
            if procedure not in day_procedures[day]:
                day_procedures[day].append(procedure)

    shortlist = []
    for day in sorted(day_procedures):
        procedures = ', '.join(day_procedures[day])
        shortlist.append(f'Day {day} procedures: {procedures}')

    return shortlist

def unique_ints(pattern, filelist):
    vals = set()
    for f in filelist:
        m = re.search(pattern, str(f))
        if m:
            vals.add(int(m.group(1)))
    return sorted(vals)


def pick_best_day_hybrid(
    data, filelist,
    E_col=6, I_col=7,
    E_star=1.98, dE=0.02,      # okno ~ [1.96, 2.00] V
    min_pts=20,
    q=0.995,
    use_abs=True
):
    """
    Score dne:
      - pokud je dost bodů v okně kolem E_star: median(|I|) v okně
      - jinak fallback: Q_q(|I|) z celého dne
    """
    day_window_vals = {}
    day_all_vals = {}

    for arr, f in zip(data, filelist):
        try:
            dnum, pnum, snum, pcode, ch = parse_mpt_filename(f)
        except Exception:
            continue

        if arr is None or len(arr) == 0:
            continue

        E = arr[:, E_col]
        I = arr[:, I_col]
        m = np.isfinite(E) & np.isfinite(I)
        if not np.any(m):
            continue

        E = E[m]
        I = I[m]
        if use_abs:
            I = np.abs(I)

        day_all_vals.setdefault(dnum, []).append(I)

        mw = np.abs(E - E_star) <= dE
        if np.any(mw):
            day_window_vals.setdefault(dnum, []).append(I[mw])

    day_to_score = {}
    for day, all_chunks in day_all_vals.items():
        Iall = np.concatenate(all_chunks) if len(all_chunks) > 1 else all_chunks[0]

        win_chunks = day_window_vals.get(day, [])
        if win_chunks:
            Iwin = np.concatenate(win_chunks) if len(win_chunks) > 1 else win_chunks[0]
        else:
            Iwin = np.array([])

        if Iwin.size >= min_pts:
            score = float(np.median(Iwin))
        else:
            score = float(np.nanquantile(Iall, q))

        day_to_score[day] = score

    if not day_to_score:
        return None, {}

    best_day = max(day_to_score, key=day_to_score.get)
    return best_day, day_to_score


def genproc(filelist):
    """
    Processes a list of filenames to extract the first number and 
    the string after the number separated by '_XXX_' or similar patterns.

    Args:
        filelist (list of str): List of filenames (e.g., "01_OCV_PEIS5").
    
    Returns:
        list of str: Extracted number and associated string for each filename.
    """
    results = []
    for filef in filelist:
        # Generalized pattern to match the first number and a subsequent capitalized string
        match = re.search(r'(\d+).*?_([A-Z]+)', filef)
        if match:
            number = match.group(1)  # Extract the first number
            label = match.group(2)  # Extract the capitalized string after the number
            results.append(f"{number}: {label}")
    return results

def loadFilesCSV(folder,wild): #folder that should be searched. wild is the constraint for the wild card (* all entries, ? only one)
    data=[] # loaded data
    filelist=[] # list of the datas in data, names with the complete paths
    filename=[] # list of the names only in the folder
    os.chdir(folder)
    qfilelist = glob.glob(wild)
    for i,ffile in enumerate(qfilelist):
        qfilelist[i] = os.path.join(folder,ffile)
    qfilelist=sorted(qfilelist)
    for n in qfilelist:
        if n[-4:] == '.csv':
            filelist.append(str(n))
            filename.append(os.path.basename(str(n)))
            scann=(pd.read_csv(str(n),usecols=[0]))
            values = list(x for x in scann["name"])
            scans=[]
            for i,val in enumerate(values):
                scan=re.findall(r'\d+', values[i])
                scans.append(int(scan[0]))
            data.append(scans)
            data.append(np.genfromtxt(str(n),comments='#', delimiter=',', skip_header=1))
    return data,filelist,filename




def loadFilesMPT(folder, wild):
    data = []
    filelist = []
    filename = []
    
    # Verify if the folder exists before changing directory
    if not os.path.exists(folder):
        raise FileNotFoundError(f"The folder {folder} does not exist.")
    
    os.chdir(folder)
    qfilelist = glob.glob(wild)
    for i, ffile in enumerate(qfilelist):
        qfilelist[i] = os.path.join(folder, ffile)
    qfilelist = sorted(qfilelist)
    
    for n in qfilelist:
        if n.endswith('.mpt'):
            # Check if the file is empty
            if os.path.getsize(n) == 0:
                continue
            
            # Open in text mode (not binary)
            with open(str(n), 'r', encoding="cp855") as f:
                f.readline()
                second_line = f.readline()
                try:
                    nb_header_lines = int(second_line.split(":")[-1].strip())  # More robust extraction
                except (IndexError, ValueError):
                    print(f"Skipping file {n} due to header parsing issue.")
                    continue  # Skip this file instead of raising an error
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    file_data = np.genfromtxt(str(n), delimiter="\t", encoding='cp855', skip_header=nb_header_lines)
                
                # Skip files with irregular data structures
                if file_data.size == 0 or len(file_data.shape) != 2:
                    continue
                
                data.append(file_data)
                filelist.append(str(n))
                filename.append(os.path.basename(str(n)))
            except ValueError:
                print(f"Skipping file {n} due to data inconsistency.")
                continue
    
    return data, filelist, filename

# opens all data from a folder which contains the wild constraint in the name.



def impedance_fit_MPT(folder, wild, ohmic, EDC, fmax, remax):
    dataEL,filelistEL,filenameEL = pyfai.loadFilesMPT(folder,wild)
    res=[]
    for i,file in enumerate(dataEL):
        filtered=[1,2]
        for j,line in enumerate(file):
            if (round(line[6],2)==EDC) and (line[0]<5000) and (line[1]<1.5):
                filtered=np.vstack([filtered,[line[1],line[2]]])
        filtered = np.delete(filtered, (0,0), axis=0)
        plt.plot(filtered[:,0],filtered[:,1])
        p = np.polyfit(filtered[:,0],filtered[:,1], 1)
        res.append(-p[1]/p[0])
    return res


def selectEL(data,filelist,nr,day = 'all',procedure='SV', loop= [1,2,3],proc_nr=[1],seq='all', plot='N',\
             dayn=[7,8],procn=[18,19],S=4.62, EIS_E='all',fR=[3E4,1]):
    firstSV=81
    lastSV=220
    llengthSV=361
    llengthCA=1200
    llengthIS=1037
    
#     plt.close("all")
    tf=np.zeros(len(data))
    if day == 'all':
        dnum = []
        for i,filef in enumerate(filelist):
            numbers = re.findall(r'\d+', filef)
            dnum.append(int(numbers[0]))
            day = set(dnum)

    if seq == 'all':
       snum = []
       for i,filef in enumerate(filelist):
           numbers = re.findall(r'\d+', filef)
           snum.append(int(numbers[-1]))
           seq = set(snum)

    for i,filef in enumerate(filelist):
        numbers = re.findall(r'\d+', filef)
        dnum =int(numbers[0])
        snum=numbers[-1]
        if (dnum in day) and (int(numbers[1]) in proc_nr) and (int(snum) in seq):
            tf[i]=1
                #print(j)
                
    tf=tf.astype(bool)

    newdata = [val for is_good, val in zip(tf, data) if is_good]
    newfiles = [val for is_good, val in zip(tf, filelist) if is_good]

    print(newfiles)
    
    if procedure=='SV':
        copied=copy.deepcopy(newdata)
        for i,file in enumerate(newdata):
            if len(newdata[i])==0:
                continue
            for j in range(0,len(newdata[i][:,5])):
                copied[i][j,7]=np.nan
                if newdata[i][j,9] in loop:
                        start=(newdata[i][j,9]-1)*llengthSV+firstSV-1
                        end=(newdata[i][j,9]-1)*llengthSV+lastSV-1
                        if j in range(int(start),int(end)):
                            copied[i][j,7]=newdata[i][j,7]

    if procedure=='CA':
        copied=copy.deepcopy(newdata)
        for i,file in enumerate(newfiles):
            if len(newdata[i])==0:
                continue
            for j in range(0,len(newdata[i][:,7])):
                copied[i][j,10]=np.nan
                for q in loop:
                        start=(q-1)*llengthCA
                        end=(q)*llengthCA-1
                        if j in range(int(start),int(end)):
                            copied[i][j,7]=copied[i][j,7]-newdata[i][int(start),7]
                            copied[i][j,10]=newdata[i][j,10]
    if procedure=='IS':
        copied=copy.deepcopy(newdata)
        for i,file in enumerate(newfiles):
            if len(newdata[i])==0:
                continue
            for j in range(0,len(newdata[i][:,0])):
                copied[i][j,2]=np.nan
                for q in loop:
                        Lfull=False
                        start=(q-1)*llengthIS
                        end=(q)*llengthIS-1
                        if (j in range(int(start)+2,int(end))) and (EIS_E=='all' or round(newdata[i][j,6],3) in EIS_E) and (newdata[i][j,2]!=0):
                            if newdata[i][j,0]<=fR[0] and newdata[i][j,0]>=fR[1]:
                                copied[i][j,2]=newdata[i][j,2]
                        
            copied[i]=copied[i][~np.isnan(copied[i]).any(axis=1),:]

    if procedure=='MB':
        copied=copy.deepcopy(newdata)
        for i,file in enumerate(newfiles):
            if len(newdata[i])==0:
                continue
            for j in range(0,len(newdata[i][:,0])):
                copied[i][j,8]=copied[i][j,8]-newdata[i][0,8]

    if plot=='Y':
        fig = plt.figure()
        ax = fig.add_subplot()
        for i in range(0,len(newdata)):
            if len(newdata[i])==0:
                continue
            elif procedure=='SV':
                ax.plot(copied[i][:,8]/(S*1000),copied[i][:,7],label='MEA'+nr+', '+newfiles[i][0:4]+', loop: '+str(loop)[1:-1])
                ax.set_ylabel(r'Cell Voltage / V')
                ax.set_xlabel(r'Current Density / A cm$^{-2}$')
            elif procedure=='CA':
                for q in loop:
                        start=(q-1)*llengthCA
                        end=(q)*llengthCA-1
                        lab='MEA'+nr+', '+newfiles[i][0:4]+', loop: '+str(loop)[1:-1]
                        ax.plot(copied[i][start:end,7]/3600,copied[i][start:end,10]/(S*1000), label= lab if lab \
            not in plt.gca().get_legend_handles_labels()[1] else '')
                ax.set_xlabel(r'Time / hours')
                ax.set_ylabel(r'Current Density / A cm$^{-2}$')
            elif procedure=='MB':
                copied=newdata
                ax.plot(copied[i][:,8]/3600,copied[i][:,11]/(S*1000),label='MEA'+nr+', '++newfiles[i][0:4])
                ax.set_xlabel(r'Time / hours')
                ax.set_ylabel(r'Current Density / A cm$^{-2}$')
            elif procedure=='IS':
                ax.set_aspect(r'equal', 'box')
                ax.plot(copied[i][:,1]*(S),copied[i][:,2]*(S),label='MEA'+nr+', '+newfiles[i][0:4]+', loop: '+str(loop)[1:-1])
                ax.set_xlabel(r'Re Z / $ \Omega$ cm$^{-2} $')
                ax.set_ylabel(r'-Im / $ \Omega$ cm$^{-2} $')
        plt.legend(frameon=False)
        plt.tight_layout()
    return copied,newfiles,loop


def parse_mpt_filename(fname):
    """
    Obecný parser pro jména typu:
      I_Day1_procedure1_03_SV.mpt
      I_Day1_procedure1_03_SV_C01.mpt
      I_Day5_procedure2_04_PEIS_C02.mpt
    Vrací: day, proc_nr, seq, proc_code, channel (channel může být None)
    """
    base = os.path.splitext(os.path.basename(fname))[0]  # bez .mpt
    parts = base.split('_')

    # Očekáváme:
    #  [0]  I
    #  [1]  DayX
    #  [2]  procedureY
    #  [3]  seq (např. 03)
    #  [4]  typ měření (SV, PEIS, CV, IS, ...)
    #  [5]  volitelné Cxx (např. C01)
    if len(parts) not in (5, 6):
        raise ValueError(f"Unexpected filename format: {base}")

    try:
        day = int(parts[1].replace('Day', ''))
        proc_nr = int(parts[2].replace('procedure', ''))
        seq = int(parts[3])
    except ValueError:
        raise ValueError(f"Cannot parse day/procedure/seq from: {base}")

    proc_code = parts[4]

    channel = None
    if len(parts) == 6 and parts[5].startswith('C'):
        try:
            channel = int(parts[5][1:])  # 'C01' -> 1
        except ValueError:
            channel = None

    return day, proc_nr, seq, proc_code, channel

def selectProc(
    data,
    filelist,
    *,
    day='all',
    procedure='SV',
    loop=[1, 2, 3],
    proc_nr=[1],
    seq='all',
    EIS_E='all',
    fR=[3E4, 1],
):
    firstSV = 81
    lastSV = 220
    llengthSV = 361
    llengthCA = 1200
    llengthIS = 1037

    # 1b) zjisti všechna proc_nr, pokud je proc_nr='all'
    if proc_nr == 'all':
        proc_nr = set()
        for filef in filelist:
            try:
                dnum, pnum, snum, pcode, ch = parse_mpt_filename(filef)
                proc_nr.add(pnum)
            except ValueError as e:
                print(e)

    # 2b) zjisti všechny loopy, pokud je loop='all'
    # (loop number isn't in filename in your code; it's in the data column file[:,9])
    if loop == 'all':
        # defer until after selection, or just set a wide range:
        loop = list(range(1, 100))

        tf = np.zeros(len(data), dtype=bool)



    # 1) zjisti všechny dny, pokud je day='all'
    if day == 'all':
        day = set()
        for filef in filelist:
            try:
                dnum, pnum, snum, pcode, ch = parse_mpt_filename(filef)
                day.add(dnum)
            except ValueError as e:
                print(e)

    # 2) zjisti všechny sekvence, pokud je seq='all'
    if seq == 'all':
        seq = set()
        for filef in filelist:
            try:
                dnum, pnum, snum, pcode, ch = parse_mpt_filename(filef)
                seq.add(snum)
            except ValueError as e:
                print(e)

    # 3) označ soubory, které odpovídají day, proc_nr, seq
    for i, filef in enumerate(filelist):
        try:
            dnum, pnum, snum, pcode, ch = parse_mpt_filename(filef)
        except ValueError as e:
            print(e)
            continue

        if (dnum in day) and (pnum in proc_nr) and (snum in seq):
            tf[i] = True

    # 4) filtruj data a soubory
    newdata = [val for is_good, val in zip(tf, data) if is_good]
    newfiles = [val for is_good, val in zip(tf, filelist) if is_good]

    print("Selected files:", newfiles)

    # 5) původní logika podle procedure
    if procedure == 'SV':
        copied = copy.deepcopy(newdata)
        for i, file in enumerate(newdata):
            if len(file) == 0:
                continue
            for j in range(len(file[:, 5])):
                copied[i][j, 7] = np.nan
                if file[j, 9] in loop:
                    start = (file[j, 9] - 1) * llengthSV + firstSV - 1
                    end = (file[j, 9] - 1) * llengthSV + lastSV - 1
                    if j in range(int(start), int(end)):
                        copied[i][j, 7] = file[j, 7]

    elif procedure == 'CA':
        copied = copy.deepcopy(newdata)
        for i, _file in enumerate(newfiles):
            if len(newdata[i]) == 0:
                continue
            for j in range(len(newdata[i][:, 7])):
                copied[i][j, 10] = np.nan
                for q in loop:
                    start = (q - 1) * llengthCA
                    end = q * llengthCA - 1
                    if j in range(int(start), int(end)):
                        copied[i][j, 7] = copied[i][j, 7] - newdata[i][int(start), 7]
                        copied[i][j, 10] = newdata[i][j, 10]

    elif procedure == 'IS':
        copied = copy.deepcopy(newdata)
        for i, _file in enumerate(newfiles):
            if len(newdata[i]) == 0:
                continue
            modified_data = []
            prev_freq = None

            for j in range(len(newdata[i][:, 0])):
                freq = newdata[i][j, 0]

                if prev_freq is not None and prev_freq > 0 and freq > 0:
                    if freq / prev_freq > 10:
                        modified_data.append([np.nan] * newdata[i].shape[1])

                row = list(newdata[i][j])
                row[2] = np.nan

                for q in loop:
                    start = (q - 1) * llengthIS
                    end = q * llengthIS - 1
                    if (
                        j in range(int(start) + 2, int(end))
                        and (EIS_E == 'all' or round(newdata[i][j, 6], 3) in EIS_E)
                        and (newdata[i][j, 2] != 0)
                    ):
                        if fR[0] >= newdata[i][j, 0] >= fR[1]:
                            row[2] = newdata[i][j, 2]

                modified_data.append(row)
                prev_freq = freq if np.isfinite(freq) and freq > 0 else None

            copied[i] = np.array(modified_data)

    elif procedure == 'MB':
        copied = copy.deepcopy(newdata)
        for i, _file in enumerate(newfiles):
            if len(newdata[i]) == 0:
                continue
            for j in range(len(newdata[i][:, 0])):
                copied[i][j, 8] = copied[i][j, 8] - newdata[i][0, 8]

    return copied, newfiles, loop

def plotData(datasets, labels, colors, areas, procedure, plot_params, folder, gradient='Y', merge=False, filenames= None):
    """
    Plots the processed data based on the specified procedure for multiple datasets.
    """
    def generate_gradient_color(base_color, num_curves, factor=0.7):
        base_color = mcolors.to_rgb(base_color)
        gradient_colors = [mcolors.to_hex((
            base_color[0] * (1 - factor * i / num_curves), 
            base_color[1] * (1 - factor * i / num_curves), 
            base_color[2] * (1 - factor * i / num_curves)
        )) for i in range(num_curves)]
        return gradient_colors

    fig, ax = plt.subplots()
    added_labels = set()

    label_order_flag = plot_params.get('label_order', 'N')

    for dataset_idx, (dataset, label, base_color, S) in enumerate(zip(datasets, labels, colors, areas)):
        num_curves = len(dataset)

        # vezmeme si seznam souborů pro daný dataset
        dataset_fnames = filenames[dataset_idx] if filenames is not None else [None] * num_curves

        # pokud chceme číslovat podle dne, spočítáme pořadí křivek (rank) podle dne
        if label_order_flag == 'Y' and filenames is not None:
            days = []
            for f in dataset_fnames:
                if f is None:
                    days.append(None)
                else:
                    dnum, pnum, snum, pcode, ch = parse_mpt_filename(f)
                    days.append(dnum)

            # indexy křivek, které mají definovaný den
            valid_idx = [i for i, d in enumerate(days) if d is not None]
            # seřadit podle dne (1,2,3,...,10,...)
            sorted_idx = sorted(valid_idx, key=lambda i: days[i])
            # mapování: index křivky -> pořadí (1..N)
            day_rank = {i: r + 1 for r, i in enumerate(sorted_idx)}
        else:
            day_rank = {}

        # gradient barev
        if gradient == 'Y':
            gradient_colors = generate_gradient_color(base_color, num_curves)
        else:
            gradient_colors = [base_color] * num_curves
        
        for curve_idx, (data, color) in enumerate(zip(dataset, gradient_colors)):
            if len(data) == 0:
                continue

            if procedure == 'MB' and len(data[:, 8]) < plot_params.get('min_size', 0):
                continue
            
            front = plot_params.get('front_gap', 0)
            back = plot_params.get('back_gap', -1)
            
            # původní legenda
            if isinstance(label, list):
                plot_label = label[curve_idx] if curve_idx < len(label) else None
            else:
                plot_label = label if label not in added_labels else None
            
            # x,y podle procedury
            if procedure == 'SV':
                x_vals = data[front:back, 8] / (S * 1000)
                y_vals = data[front:back, 7]
                ax.plot(x_vals, y_vals, label=plot_label, color=color,
                        linewidth=plot_params.get('linewidth', 1))

            elif procedure == 'CA':
                x_vals = data[front:back, 7] / 3600
                y_vals = data[front:back, 10] / (S * 1000)
                ax.plot(x_vals, y_vals, label=plot_label, color=color,
                        linewidth=plot_params.get('linewidth', 1))

            elif procedure == 'IS':
                ax.set_aspect('equal', 'box')
                x_vals = data[front:back, 1] * S
                y_vals = data[front:back, 2] * S
                ax.plot(x_vals, y_vals, label=plot_label, color=color,
                        linewidth=plot_params.get('linewidth', 1))

            elif procedure == 'MB':
                x_vals = data[front:back, 8] / 3600
                y_vals = data[front:back, 11] / (S * 1000)
                ax.plot(x_vals, y_vals, label=plot_label, color=color,
                        linewidth=plot_params.get('linewidth', 1))
            
            if plot_label and plot_label not in added_labels:
                added_labels.add(plot_label)

            # 🔢 číslo křivky podle dne
            if label_order_flag == 'Y' and day_rank:
                if curve_idx in day_rank:
                    num = day_rank[curve_idx]  # 1 = nejmenší den, 2 = druhý, ...
                    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
                    if np.any(valid):
                        idx = np.where(valid)[0][-1]
                        dx = 0.01 * (np.nanmax(x_vals) - np.nanmin(x_vals) + 1e-12)
                        dy = 0.01 * (np.nanmax(y_vals) - np.nanmin(y_vals) + 1e-12)

                        ax.text(
                            x_vals[idx] + dx,
                            y_vals[idx] + dy,
                            str(days[curve_idx]),
                            fontsize=plot_params.get('curve_label_fontsize',
                                                     plot_params.get('tick_fontsize', 12)),
                            ha='left',
                            va='center'
                        )        
        # popisky os
        if procedure == 'SV':
            ax.set_ylabel(r'Cell Voltage / V', fontsize=plot_params.get('axis_label_fontsize', 14))
            ax.set_xlabel(r'Current Density / A cm$^{-2}$', fontsize=plot_params.get('axis_label_fontsize', 14))
        elif procedure == 'CA':
            ax.set_xlabel(r'Time / hours', fontsize=plot_params.get('axis_label_fontsize', 14))
            ax.set_ylabel(r'Current Density / A cm$^{-2}$', fontsize=plot_params.get('axis_label_fontsize', 14))
        elif procedure == 'IS':
            ax.set_xlabel(r'Re Z / $\Omega$ cm$^{-2}$', fontsize=plot_params.get('axis_label_fontsize', 14))
            ax.set_ylabel(r'-Im / $\Omega$ cm$^{-2}$', fontsize=plot_params.get('axis_label_fontsize', 14))
        elif procedure == 'MB':
            ax.set_xlabel(r'Time / hours', fontsize=plot_params.get('axis_label_fontsize', 14))
            ax.set_ylabel(r'Current Density / A cm$^{-2}$', fontsize=plot_params.get('axis_label_fontsize', 14))
    
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    plt.legend(by_label.values(), by_label.keys(),
               frameon=False,
               fontsize=plot_params.get('legend_fontsize', 12))
    plt.xticks(fontsize=plot_params.get('tick_fontsize', 12))
    plt.yticks(fontsize=plot_params.get('tick_fontsize', 12))

    if 'x_range' in plot_params:
        ax.set_xlim(plot_params['x_range'])
    if 'y_range' in plot_params:
        ax.set_ylim(plot_params['y_range'])
    if 'x_ticks' in plot_params:
        ax.set_xticks(plot_params['x_ticks'])
    if 'y_ticks' in plot_params:
        ax.set_yticks(plot_params['y_ticks'])

    plt.tight_layout()
    plt.savefig(f"{folder}_{procedure}.png", transparent=True, bbox_inches="tight")
    plt.show()





def selectProcRDE(data, filelist, procedure='all', proc_nr='all', seq='all', repetition='all', cycle_range='all', EIS_E='all', fR='all', U_shift_to_RHE = 0):
    """
    Processes and filters RDE data based on specified criteria.

    Parameters:
        data (list): List of numpy arrays containing RDE data.
        filelist (list): List of filenames corresponding to the data.
        procedure (str): Procedure name to filter (e.g., 'CA', 'LSV', 'CV', 'PEIS', or 'all').
        proc_nr (list or str): List of procedure numbers to filter, or 'all' to include all.
        seq (list or str): List of sequence numbers to filter, or 'all' to include all.
        repetition (list or str): List of repetition numbers to filter, or 'all' to include all.
        cycle_range (list, tuple, or str): Range of cycles to include, or 'all' to include all.
        EIS_E (list or str): Voltage values to filter, or 'all' to include all.
        fR (list or str): Frequency values to filter, or 'all' to include all.

    Returns:
        tuple: Filtered data, filenames, and a list of selected cycles per file.
    """

    def expand_cycle_range(cycle_range):
        """
        Expands a mixed list of integers and tuples into a flat list of integers.
    
        Parameters:
            cycle_range (list): List containing integers or tuples representing ranges.
                                Example: [1, (3, 5)] expands to [1, 3, 4, 5].
    
        Returns:
            list: A flat list of integers representing all selected cycles.
        """
        expanded_cycles = []
        for item in cycle_range:
            if isinstance(item, tuple) and len(item) == 2:  # Check if it's a valid range tuple
                expanded_cycles.extend(range(item[0], item[1] + 1))
            elif isinstance(item, int):  # Single cycle number
                expanded_cycles.append(item)
        return list(set(expanded_cycles))  # Ensure unique values and return as a list

    
    def parse_filename(filename):
        """Parses the filename to extract proc_nr, sequence, procedure, and repetition."""
        proc_nr_match = re.match(r'^(\d+)', filename)
        proc_nr = int(proc_nr_match.group(1)) if proc_nr_match else None

        # Extract procedure name
        procedure_match = re.search(r'_([A-Za-z]+)(?:_C\d+)?(?:\d+)?(?:\.mpt)?$', filename)
        procedure_name = procedure_match.group(1) if procedure_match else None

        # Extract sequence and repetition
        seq_match = re.search(r'_(\d+)_{}|_(\d+){}'.format(procedure_name, procedure_name), filename)
        sequence = int(seq_match.group(1)) if seq_match else None
        repetition_match = re.search(r'{}(\d+)(?!.*C\d+)'.format(procedure_name), filename)
        repetition = int(repetition_match.group(1)) if repetition_match else None

        return proc_nr, sequence, procedure_name, repetition

    # Handle 'all' defaults for seq, proc_nr, and repetition
    if seq == 'all' or seq is None:
        seq = {parse_filename(filef)[1] for filef in filelist if parse_filename(filef)[1] is not None} | {None}
    if proc_nr == 'all' or proc_nr is None:
        proc_nr = {parse_filename(filef)[0] for filef in filelist if parse_filename(filef)[0] is not None} | {None}
    if repetition == 'all' or repetition is None:
        repetition = {parse_filename(filef)[3] for filef in filelist if parse_filename(filef)[3] is not None} | {None}

    # Initialize boolean filter
    tf = np.zeros(len(data), dtype=bool)

    # Filter filelist and data based on sequence, procedure number, repetition, and procedure name
    for i, filef in enumerate(filelist):
        proc_nr_file, sequence, proc_name, repetition_file = parse_filename(filef)


        # Check filtering conditions
        if procedure != 'all' and proc_name != procedure:
            continue
        if proc_nr_file not in proc_nr:
            continue
        if sequence not in seq:
            continue
        if repetition_file not in repetition:
            continue

        tf[i] = True

    # Filter the data and file list based on the boolean array tf
    newdata = [val for is_good, val in zip(tf, data) if is_good]
    newfiles = [val for is_good, val in zip(tf, filelist) if is_good]

    
    # Process data based on procedure
    copied = []
    cycles_per_file = []  # To store the selected cycles for each file

    for file in newdata:
        if len(file) == 0:
            copied.append(np.empty((0, file.shape[1])))  # Empty array with the same number of columns
            cycles_per_file.append([])  # Empty list for cycles
            continue

        elif procedure == 'CA':
            cycle_column = file[:, -3]  # Third-to-last column for cycle numbers
            
            # Filter rows by cycle number range
            if cycle_range != 'all':
                expanded_cycle_range = expand_cycle_range(cycle_range)  # Expand ranges into discrete values               
                cycle_mask = np.isin(cycle_column, expanded_cycle_range)
                filtered_file = file[cycle_mask]
                selected_cycles = np.unique(cycle_column[cycle_mask])
            else:
                filtered_file = file  # Include all cycles
                selected_cycles = np.unique(cycle_column)



        elif procedure == 'CV':
            # Use the 10th column (index 9) for cycle numbers
            cycle_column = file[:, 9]
        
            # Filter rows by cycle number range
            if cycle_range != 'all':
                expanded_cycle_range = expand_cycle_range(cycle_range)  # Expand ranges into discrete values
                cycle_mask = np.isin(cycle_column, expanded_cycle_range)
                filtered_file = file[cycle_mask]
                selected_cycles = np.unique(cycle_column[cycle_mask])
            else:
                filtered_file = file  # Include all cycles
                selected_cycles = np.unique(cycle_column)
        
        elif procedure == 'LSV':
            # Append the cycle column based on time jumps > 100 seconds
            time_column = file[:, 4]  # Fifth column represents time
            time_diff = np.diff(time_column, prepend=time_column[0])  # Compute time differences
        
            # Identify cycle boundaries based on time jumps > 100 seconds
            cycle_starts = np.where(time_diff > 100)[0]
            cycle_starts = np.insert(cycle_starts, 0, 0)  # Include the first cycle start
            cycle_indices = np.append(cycle_starts, len(time_column))  # Include the end of the last cycle
        
            # Assign cycle numbers to rows
            cycle_column = np.zeros(len(time_column), dtype=int)
            for idx, (start, end) in enumerate(zip(cycle_indices[:-1], cycle_indices[1:])):
                cycle_column[start:end] = idx + 1  # Assign cycle numbers starting from 1
        
            # Append the cycle column to the file
            file = np.hstack((file, cycle_column[:, None]))
        
            # Use the last column as the cycle column
            cycle_column = file[:, -1]
        
            # Filter rows by cycle number range
            if cycle_range != 'all':
                expanded_cycle_range = expand_cycle_range(cycle_range)  # Expand ranges into discrete values
                cycle_mask = np.isin(cycle_column, expanded_cycle_range)
                filtered_file = file[cycle_mask]
                selected_cycles = np.unique(cycle_column[cycle_mask])
            else:
                filtered_file = file  # Include all cycles
                selected_cycles = np.unique(cycle_column)
        
        elif procedure == 'PEIS':
            freq_column = file[:, 0]  # Assume the 1st column contains frequency
            voltage_column = file[:, 6]  # Assume the 7th column contains voltage
            cycle_column = file[:, 10]  # Assume the 11th column contains cycle numbers
        
            # Adjust voltages to RHE
            voltage_column_rhe = voltage_column + U_shift_to_RHE  # Apply RHE shift
        
            # Create masks for filtering
            voltage_mask = (np.isclose(voltage_column_rhe, EIS_E, atol=0.001)
                            if EIS_E != 'all'
                            else np.ones_like(voltage_column, dtype=bool))
            freq_mask = ((freq_column >= fR[1]) & (freq_column <= fR[0])
                         if fR != 'all'
                         else np.ones_like(freq_column, dtype=bool))
            cycle_mask = (np.isin(cycle_column, cycle_range)
                          if cycle_range != 'all' else np.ones_like(cycle_column, dtype=bool))
            # Combine masks
            combined_mask = voltage_mask & freq_mask & cycle_mask
        
            # Apply combined mask
            filtered_file = file[combined_mask]
            selected_cycles = np.unique(cycle_column[combined_mask]) if cycle_column is not None else []
        
        else:
            # For other procedures, handle cycles as usual
            cycle_column = None  # Default to no cycle column
            if procedure == 'LSV':
                cycle_column = file[:, -1]  # Last column for LSV
            elif procedure == 'CV':
                cycle_column = file[:, 9]  # 10th column for CV
            elif procedure == 'CA':
                cycle_column = file[:, 8]  # 9th column for CA
            elif procedure == 'MB':
                cycle_column = file[:, 11]  # 12th column for MB
        
            # Filter rows by cycle number range
            if cycle_range != 'all' and cycle_column is not None:
                expanded_cycle_range = expand_cycle_range(cycle_range)  # Expand ranges into discrete values
                cycle_mask = np.isin(cycle_column, expanded_cycle_range)
                filtered_file = file[cycle_mask]
                selected_cycles = np.unique(cycle_column[cycle_mask])
            else:
                filtered_file = file  # Include all cycles
                selected_cycles = np.unique(cycle_column) if cycle_column is not None else []
        
        # Add filtered data to output lists
        copied.append(filtered_file)
        cycles_per_file.append(selected_cycles.tolist())


            

    return copied, newfiles, cycles_per_file



def plotDataRDE(datasets, labels, colors, areas, base_params, plot_params, user_params_list, folder, gradient='Y', tafel_regression_results=None):
    """
    Plots the processed data based on the specified procedure for multiple datasets, with optional Tafel regression fits.
    """
    def generate_gradient_color(base_color, num_curves, factor=0.5, min_lightness=0.2):
        """
        Generate a gradient of colors starting from the base_color, avoiding excessive darkening.
        """
        base_color = np.array(mcolors.to_rgb(base_color))  # Convert base color to RGB
        gradient_colors = [
            mcolors.to_hex(base_color * max(1 - factor * i / max(1, num_curves - 1), min_lightness))
            for i in range(num_curves)
        ]
        return gradient_colors

    fig, ax = plt.subplots()
    added_labels = set()
    procedure = base_params['procedure']

    for dataset_idx, (dataset, label, base_color, S) in enumerate(zip(datasets, labels, colors, areas)):
        if procedure in ['LSV', 'CV']:
            # Count unique cycles across the dataset for gradient generation
            all_cycles = [cycle for data in dataset if len(data) > 0 for cycle in np.unique(data[:, -1 if procedure == 'LSV' else 9])]
            unique_cycles = np.unique(all_cycles)
            num_curves = len(unique_cycles)
        else:
            # Default to the dataset length for non-cycle-based procedures
            num_curves = len(dataset)
            unique_cycles = []  # Placeholder for non-LSV/CV procedures
                
        # Generate gradient colors
        if gradient == 'Y' and num_curves > 1:
            gradient_colors = generate_gradient_color(base_color, num_curves)
            cycle_to_color = {cycle: gradient_colors[idx] for idx, cycle in enumerate(unique_cycles)}
        else:
            gradient_colors = [base_color] * num_curves
            cycle_to_color = {cycle: base_color for cycle in unique_cycles}
        curve_nr=0
        for curve_idx, data in enumerate(dataset):
            curve_nr+=1
            if len(data) == 0:
                continue
            front = plot_params.get('front_gap', 0)
            back = plot_params.get('back_gap', -1)
            
            # Handle label assignment safely for all procedures
            if isinstance(label, list):
                plot_label = label[curve_idx] if curve_idx < len(label) else None
            else:
                plot_label = label if label not in added_labels else None

            # For LSV and CV, define unique_cycles to control plotting logic
            if procedure in ['LSV', 'CV']:
                cycle_column = data[:, -1 if procedure == 'LSV' else 9]  # Extract cycle column
                unique_cycles = np.unique(cycle_column)  # Get unique cycle numbers
                # Determine if we are plotting in normal or Tafel mode
                is_tafel = tafel_regression_results is not None
                voltage_col, current_col, cycle_col = (6, 7, -1) if procedure == 'LSV' else (7, 8, 9)
                cycle_column = data[:, cycle_col]
                unique_cycles_in_data = np.unique(cycle_column)

                if is_tafel:
                    unique_cycles_in_data = np.unique(cycle_column)  # Get actual cycle numbers
                
                    # Ensure we generate enough colors for all dataset curves
                    num_cycles_in_data = len(dataset)
                    gradient_colors = generate_gradient_color(base_color, num_cycles_in_data)
                
                    # Compute relative cycle positions
                    cycle_relative_index = {cycle: idx for idx, cycle in enumerate(sorted(unique_cycles_in_data))}
                
                    # Assign colors using the relative index
                    cycle_to_color = {
                        cycle: gradient_colors[curve_nr-1]
                        for cycle in unique_cycles_in_data
                    }


                    
                    for cycle in unique_cycles_in_data:
                        color = cycle_to_color.get(cycle, base_color)  # Assign color for this cycle
                        cycle_mask = cycle_column == cycle
                        cycle_data = data[cycle_mask]
                
                        # Append NaN to break the line between cycles
                        voltage_with_nan = np.append(cycle_data[:, voltage_col], np.nan)
                        log_current_with_nan = np.append(cycle_data[:, current_col], np.nan)
                        print(color)
                        ax.plot(
                            voltage_with_nan,
                            log_current_with_nan,
                            label=plot_label if cycle == unique_cycles_in_data[0] else None,
                            color=color,
                            linewidth=plot_params.get('linewidth', 1)
                        )
                
                    # Add regression line
                    for result in tafel_regression_results[dataset_idx]['results']:
                        if result['cycle'] in unique_cycles_in_data:
                            slope = result.get('slope')
                            intercept = result.get('intercept')
                
                            if slope is None or intercept is None:
                                continue
                
                            tafel_range = user_params_list[dataset_idx].get('Tafel_ranges', None)
                            if tafel_range:
                                cycle_index = int(result['cycle'] - 1)
                                if cycle_index < len(tafel_range):
                                    start, end = tafel_range[cycle_index]
                                else:
                                    start, end = tafel_range[-1]
                
                                extended_start = start - 0.5 * (end - start)
                                extended_end = end + 0.5 * (end - start)
                
                                overpotential_range = data[front:back, voltage_col]
                                mask = (overpotential_range >= extended_start) & (overpotential_range <= extended_end)
                                limited_overpotential_range = overpotential_range[mask]
                
                                if len(limited_overpotential_range) > 1:
                                    fit_line = 1000 * limited_overpotential_range / slope + intercept
                                    ax.plot(
                                        limited_overpotential_range,
                                        fit_line,
                                        linestyle='--',
                                        color=cycle_to_color.get(result['cycle'], base_color),  # Use the same color as the cycle
                                        linewidth=0.8
                                    )

                else:
                    for cycle in unique_cycles_in_data:
                        color = cycle_to_color.get(cycle, base_color)  # Assign color for this cycle
                        cycle_mask = cycle_column == cycle
                        cycle_data = data[cycle_mask]

                        ax.plot(
                            cycle_data[front:back, voltage_col],
                            cycle_data[front:back, current_col] / S,
                            label=plot_label,
                            color=color,
                            linewidth=plot_params.get('linewidth', 1)
                        )

            elif procedure == 'CA':
                color = base_color  # Assign color for this cycle
                # Identify time gaps > 100 seconds and insert NaN values
                time_column = data[front:back, 7] / 3600  # Convert time to hours
                current_column = data[front:back, 10] / S  # Normalize current density
                time_diff = np.diff(time_column, prepend=time_column[0])
                
                time_with_nan = []
                current_with_nan = []
                
                for i in range(len(time_column)):
                    if i > 0 and time_diff[i] > (100 / 3600):  # If time gap > 100s
                        time_with_nan.append(np.nan)  # Insert NaN for time gap
                        current_with_nan.append(np.nan)  # Insert NaN for current gap
                    time_with_nan.append(time_column[i])
                    current_with_nan.append(current_column[i])

                time_with_nan = np.array(time_with_nan, dtype=float)
                current_with_nan = np.array(current_with_nan, dtype=float)
                
                ax.plot(
                    np.array(time_with_nan, dtype=float), 
                    np.array(current_with_nan, dtype=float), 
                    label=plot_label, 
                    color=color, 
                    linewidth=plot_params.get('linewidth', 1)
                )

            elif procedure == 'PEIS':
                color = base_color  # Assign color for this cycle
                # Insert NaN values whenever the cycle number changes
                real_col = data[front:back, 1]  # Real impedance
                imag_col = data[front:back, 2]  # Imaginary impedance
                cycle_column = data[front:back, 10]  # Cycle column

                real_with_nan = []
                imag_with_nan = []

                prev_cycle = cycle_column[0]
                for i in range(len(real_col)):
                    real_with_nan.append(real_col[i])
                    imag_with_nan.append(imag_col[i])
                    if cycle_column[i] != prev_cycle:  # If cycle changes
                        real_with_nan.append(np.nan)
                        imag_with_nan.append(np.nan)
                    prev_cycle = cycle_column[i]

                ax.set_aspect('equal', 'box')
                ax.plot(real_with_nan, imag_with_nan, label=plot_label, color=color, linewidth=plot_params.get('linewidth', 1))

            if plot_label and plot_label not in added_labels:
                added_labels.add(plot_label)


    if tafel_regression_results and procedure in ['LSV', 'CV']:
        ax.set_xlabel(r'Overpotential / V', fontsize=plot_params.get('axis_label_fontsize', 14))
        ax.set_ylabel(r'log(Current Density / mA cm$^{-2})$', fontsize=plot_params.get('axis_label_fontsize', 14))
    elif not tafel_regression_results and procedure in ['LSV', 'CV']:
        ax.set_xlabel(r'Voltage / V$_{RHE}$', fontsize=plot_params.get('axis_label_fontsize', 14))
        ax.set_ylabel(r'Current Density / mA cm$^{-2}$', fontsize=plot_params.get('axis_label_fontsize', 14))
    elif procedure == 'CA':
        ax.set_xlabel(r'Time / hours', fontsize=plot_params.get('axis_label_fontsize', 14))
        ax.set_ylabel(r'Current Density / mA cm$^{-2}$', fontsize=plot_params.get('axis_label_fontsize', 14))
    elif procedure == 'PEIS':
        ax.set_xlabel(r'Re Z / $\Omega$ cm$^{-2}$', fontsize=plot_params.get('axis_label_fontsize', 14))
        ax.set_ylabel(r'-Im Z / $\Omega$ cm$^{-2}$', fontsize=plot_params.get('axis_label_fontsize', 14))

    handles, legend_labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    plt.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=plot_params.get('legend_fontsize', 12))
    plt.xticks(fontsize=plot_params.get('tick_fontsize', 12))
    plt.yticks(fontsize=plot_params.get('tick_fontsize', 12))

    if 'x_range' in plot_params:
        ax.set_xlim(plot_params['x_range'])
    if 'y_range' in plot_params:
        ax.set_ylim(plot_params['y_range'])
    if 'x_ticks' in plot_params:
        ax.set_xticks(plot_params['x_ticks'])
    if 'y_ticks' in plot_params:
        ax.set_yticks(plot_params['y_ticks'])


    plt.tight_layout()
    plt.savefig(f"{folder}_{procedure}.png", transparent=True, bbox_inches="tight")
    plt.show()









def correct_R(data_processed, user_params_list, base_params):
    """
    Applies ohmic resistance corrections to the filtered data based on user-defined parameters.
    For LSV, splits data into separate curves based on the cycle number column.

    Parameters:
        data_processed (list of list of np.ndarray): Processed data from selectProcRDE.
        user_params_list (list of dict): User parameters containing ohmic resistances ('R_ohm').
        base_params (dict): Base parameters containing correction options ('Correct_by_R' or 'Correct_to_R').

    Returns:
        list of list of np.ndarray: Corrected data with the same structure as the input.
    """
    corrected_data = []

    # Determine the correction mode
    correct_by_R = base_params.get('Correct_by_R', None)
    correct_to_R = base_params.get('Correct_to_R', None)
    RHE_shift = base_params.get('U_shift_to_RHE', 0)

    if correct_by_R is None and correct_to_R is None:
        # Return the original data without any modifications
        return data_processed

    if correct_by_R and correct_to_R:
        raise ValueError("Both 'Correct_by_R' and 'Correct_to_R' cannot be active simultaneously. Choose one.")

    for dataset, user_params in zip(data_processed, user_params_list):
        R_ohm = user_params.get('R_ohm', [])
        
        # Extend R_ohm list if fewer values are provided than curves
        if len(R_ohm) < len(dataset):
            R_ohm.extend([R_ohm[-1]] * (len(dataset) - len(R_ohm)))

        corrected_dataset = []
        for curve, R in zip(dataset, R_ohm):
            # Skip empty or invalid data
            if len(curve) == 0 or curve.ndim != 2:
                corrected_dataset.append(curve)  # Append as-is
                continue

            # Create a copy of the entire dataset
            corrected_curve = curve.copy()

            # Determine the effective resistance
            if correct_by_R:
                R_final = R * correct_by_R
            elif correct_to_R:
                R_final = R - correct_to_R
            else:
                R_final = R  # No correction applied
            
            # Apply correction based on the procedure
            if base_params['procedure'] == 'LSV':
                # Split the curve into segments based on cycle numbers
                cycle_numbers = corrected_curve[:, -1]  # Extract the cycle number column
                unique_cycles = np.unique(cycle_numbers)
        	# Extend R_ohm list if fewer values are provided than curves
                if len(R_ohm) < len(unique_cycles):
                    R_ohm.extend([R_ohm[-1]] * (len(unique_cycles) - len(R_ohm)))
                for i,cycle in enumerate(unique_cycles):
                    # Mask rows for the current cycle
                    cycle_mask = cycle_numbers == cycle
                    cycle_data = corrected_curve[cycle_mask]
                    # Determine the effective resistance
                    if correct_by_R:
                       R_final = R_ohm[i] * correct_by_R
                    elif correct_to_R:
                       R_final = R_ohm[i] - correct_to_R
                    else:
                       R_final = R_ohm[i]  # No correction applied
                    
                    # Apply ohmic correction for this cycle
                    cycle_data[:, 6] = cycle_data[:, 6] - R_final * cycle_data[:, 7] / 1000 +RHE_shift
                    
                    # Append the corrected cycle back
                    corrected_dataset.append(cycle_data)
            elif base_params['procedure'] == 'CA':
                corrected_curve[:, 10] = corrected_curve[:, 10]
                corrected_dataset.append(corrected_curve)
            elif base_params['procedure'] == 'CV':
                corrected_curve[:, 7] = corrected_curve[:, 7] - R_final * corrected_curve[:, 8] / 1000 +RHE_shift
                corrected_dataset.append(corrected_curve)
            elif base_params['procedure'] == 'PEIS' or base_params['procedure'] == 'IS':
                corrected_curve[:, 6] = corrected_curve[:, 6] - R_final * corrected_curve[:, 7] / 1000 +RHE_shift
                corrected_dataset.append(corrected_curve)
            elif base_params['procedure'] == 'SV':
                corrected_curve[:, 7] = corrected_curve[:, 7] - R_final * corrected_curve[:, 8] / 1000 +RHE_shift
                corrected_dataset.append(corrected_curve)
            elif base_params['procedure'] == 'MB':
                corrected_curve[:, 10] = corrected_curve[:, 10]
                corrected_dataset.append(corrected_curve)
                

        corrected_data.append(corrected_dataset)

    return corrected_data


def tafel_transform(data_processed, base_params, areas):
    """
    Applies Tafel transformation to voltage and normalized current density.

    Parameters:
        data_processed (list of list of np.ndarray): Processed data from `selectProcRDE` and corrected by `correct_R`.
        base_params (dict): Base parameters containing `Tafel` (bool) and `E_rev` (float).
        areas (list of float): Surface areas corresponding to each sample.

    Returns:
        list of list of np.ndarray: Transformed data for Tafel analysis if `Tafel` is True; original data otherwise.
    """
    # Check if Tafel transformation is enabled
    if not base_params.get('Tafel', False):
        return data_processed  # Return the original data if Tafel is not enabled

    E_rev = base_params.get('E_rev', 0)  # Default E_rev to 0 if not provided
    procedure = base_params.get('procedure', 'LSV')  # Default procedure to 'LSV'
    tafel_data = []

    # Set voltage and current column indices based on the procedure
    if procedure == 'LSV':
        voltage_col, current_col, cycle_col = 6, 7, -1  # LSV column indices
    elif procedure == 'CV':
        voltage_col, current_col, cycle_col = 7, 8, 9  # CV column indices
    else:
        raise ValueError(f"Unsupported procedure: {procedure}")

    for dataset, S in zip(data_processed, areas):  # Iterate through each dataset with its area
        transformed_dataset = []
        for curve in dataset:  # Iterate through each curve in the sample
            # Skip empty or invalid curves
            if len(curve) == 0 or curve.ndim != 2:
                transformed_dataset.append(curve)  # Append as-is
                continue

            # Create a copy of the curve for transformation
            transformed_curve = curve.copy()

            # Apply Tafel transformations
            # Voltage to Overpotential
            transformed_curve[:, voltage_col] = transformed_curve[:, voltage_col] - E_rev  # Overpotential = Voltage - E_rev

            # Current Density to log(Current Density)
            current_density = transformed_curve[:, current_col] / S  # Normalize by area
            with np.errstate(divide='ignore', invalid='ignore'):
                log_current_density = np.log10(np.abs(current_density))  # log10 of absolute value
                log_current_density[current_density <= 0] = np.nan  # Set log for invalid values to NaN

            transformed_curve[:, current_col] = log_current_density  # Replace column with log(current density)
            transformed_curve[:, cycle_col] = curve[:, cycle_col]  # Copy cycle numbers explicitly

            if procedure == 'CV':
                # Subsample the data every 10 points
                subsample_indices = np.arange(0, len(transformed_curve), 10)
                subsampled_curve = transformed_curve[subsample_indices]
            
                # Compute voltage differences for the subsampled data
                voltage_diff = np.diff(subsampled_curve[:, voltage_col], prepend=subsampled_curve[0, voltage_col])
                rising_mask = voltage_diff > 0  # Identify rows with increasing voltage
            
                # Filter the original curve based on the rising trend in the subsampled data
                # Expand the subsampled rising mask to cover the original curve
                rising_indices = subsample_indices[rising_mask]
                filtered_indices = np.hstack([
                    np.arange(start, min(start + 10, len(transformed_curve))) for start in rising_indices
                ])
            
                # Apply the mask to the original curve
                transformed_curve = transformed_curve[filtered_indices]
            
            # Append the transformed curve
            transformed_dataset.append(transformed_curve)

        tafel_data.append(transformed_dataset)


    return tafel_data


def calculate_tafel_slopes(tafel_data, corrected_data, user_params_list, base_params):
    """
    Perform linear regression on specified overpotential ranges for Tafel data.

    Parameters:
        tafel_data (list of list of np.ndarray): Transformed Tafel data for each sample and cycle.
        corrected_data (list of list of np.ndarray): Corrected raw data for each sample and cycle.
        user_params_list (list of dict): List of user parameters, including `Tafel_ranges`.
        base_params (dict): Base parameters, including `procedure`.

    Returns:
        dict: Regression results with slope (mV/dec), intercept, R², exchange current density, 
              and overpotential at 10 mA/cm² for each cycle.
    """
    regression_results = []
    E_rev = base_params.get('E_rev', 0)  # Default E_rev to 0 if not provided
    overpotential_limit=base_params.get('overpotential', 10)
    overpotential_key = f"overpotential_{overpotential_limit}_mA_cm2"
    # Determine columns based on the procedure
    if base_params['procedure'] == 'LSV':
        voltage_col, current_col, cycle_col = 6, 7, -1  # LSV column indices
    elif base_params['procedure'] == 'CV':
        voltage_col, current_col, cycle_col = 7, 8, 9  # CV column indices
    else:
        raise ValueError(f"Unsupported procedure: {base_params['procedure']}")

    for sample_idx, (tafel_sample, corrected_sample, params) in enumerate(zip(tafel_data, corrected_data, user_params_list)):
        tafel_ranges = params.get('Tafel_ranges', [])
        areas = params.get('S', 1)  # Normalization area

        sample_results = []

        for tafel_curve, corrected_curve in zip(tafel_sample, corrected_sample):
            if len(tafel_curve) == 0 or tafel_curve.ndim != 2 or len(corrected_curve) == 0:
                continue  # Skip empty or invalid curves

            # Extract unique cycle numbers from the cycle column (exclude NaN values)
            unique_cycles = np.unique(tafel_curve[~np.isnan(tafel_curve[:, cycle_col]), cycle_col])

            for cycle_num in unique_cycles:
                # Filter rows corresponding to the current cycle
                cycle_mask_tafel = tafel_curve[:, cycle_col] == cycle_num
                tafel_cycle_data = tafel_curve[cycle_mask_tafel]

                cycle_mask_corrected = corrected_curve[:, cycle_col] == cycle_num
                corrected_cycle_data = corrected_curve[cycle_mask_corrected]

                if len(tafel_cycle_data) < 2 or len(corrected_cycle_data) < 2:
                    # Not enough data points for regression
                    sample_results.append({
                        'cycle': cycle_num,
                        'slope': None,
                        'intercept': None,
                        'r_squared': None,
                        'exchange_current_density': None,
                        overpotential_key: None
                    })
                    continue

                # Retrieve the Tafel range for this cycle
                cycle_index = int(cycle_num - 1)  # Convert cycle to index
                if cycle_index < len(tafel_ranges):
                    tafel_range = tafel_ranges[cycle_index]
                else:
                    tafel_range = tafel_ranges[-1]  # Use the last range as a fallback

                # Filter data within the specified Tafel range for regression
                tafel_overpotential = tafel_cycle_data[:, voltage_col]
                tafel_log_current_density = tafel_cycle_data[:, current_col]
                tafel_mask = (tafel_overpotential >= tafel_range[0]) & (tafel_overpotential <= tafel_range[1])
                filtered_overpotential = tafel_overpotential[tafel_mask]
                filtered_log_current_density = tafel_log_current_density[tafel_mask]

                # Remove NaN values from both arrays
                valid_mask = ~np.isnan(filtered_log_current_density) & ~np.isnan(filtered_overpotential)
                filtered_overpotential = filtered_overpotential[valid_mask]
                filtered_log_current_density = filtered_log_current_density[valid_mask]

                if len(filtered_overpotential) < 2:  # Require at least two points for regression
                    sample_results.append({
                        'cycle': cycle_num,
                        'slope': None,
                        'intercept': None,
                        'r_squared': None,
                        'exchange_current_density': None,
                        overpotential_key: None
                    })
                    continue

                # Perform linear regression
                slope, intercept, r_value, _, _ = linregress(filtered_overpotential, filtered_log_current_density)

                # Calculate Tafel slope (A) and exchange current density (i_0)
                try:
                    tafel_slope_mV_dec = 1000 / slope  # Convert to mV/dec
                    exchange_current_density = 10 ** intercept
                except ZeroDivisionError:
                    tafel_slope_mV_dec = None
                    exchange_current_density = None

                # Find the overpotential corresponding to 10 mA/cm² in corrected data
                current_density = corrected_cycle_data[:, current_col] / areas
                overpotential = corrected_cycle_data[:, voltage_col]
                target_mask = current_density >= overpotential_limit  # Check for current density >= 10 mA/cm²

                overpotential_10_mA_cm2 = None
                if np.any(target_mask):
                    matching_indices = np.where(target_mask)[0]  # Get indices of valid points
                    
                    if len(matching_indices) > 0:
                        num_values = min(5, len(matching_indices))  # Use up to 5 points if available
                        selected_indices = matching_indices[:num_values]  # Select first 5 values (or fewer if not enough)
                    
                        # Compute average over selected indices
                        overpotential_10_mA_cm2 = 1000 * np.mean(overpotential[selected_indices] - E_rev)
                    else:
                        overpotential_10_mA_cm2 = np.nan  # Use np.nan if no valid points
                else:
                    overpotential_10_mA_cm2 = np.nan

                # Save regression results
                sample_results.append({
                    'cycle': cycle_num,
                    'slope': tafel_slope_mV_dec,  # Tafel slope in mV/dec
                    'intercept': intercept,
                    'r_squared': r_value**2,  # R-squared
                    'exchange_current_density': exchange_current_density,
                    overpotential_key: overpotential_10_mA_cm2  # Overpotential at 10 mA/cm²
                })

        regression_results.append({
            'sample_label': params.get('label', f"Sample {sample_idx + 1}"),
            'results': sample_results
        })

    return regression_results




def save_tafel_results_to_csv(tafel_regression_results, output_file, base_params):
    """
    Saves Tafel regression results to a CSV file in an Excel-style format.

    Parameters:
        tafel_regression_results (list of dict): Tafel regression results containing slope, intercept, r-squared values, exchange current density, and overpotential.
        output_file (str): Path to the output CSV file.
    """
    overpotential_limit = base_params.get('overpotential', 10)
    overpotential_label = f'Overpotential at {overpotential_limit} mA/cm² (mV)'
    overpotential_key = f"overpotential_{overpotential_limit}_mA_cm2"
    with open(output_file, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Sample', 
            'Cycle', 
            'Slope (mV/dec)', 
            'Intercept', 
            'R-squared', 
            'Exchange Current Density (A/cm²)', 
            overpotential_label
        ])
        
        for sample in tafel_regression_results:
            sample_label = sample['sample_label']
            
            for result in sample['results']:
                cycle = result.get('cycle', 'N/A')
                slope = result.get('slope', 'N/A')
                intercept = result.get('intercept', 'N/A')
                r_squared = result.get('r_squared', 'N/A')
                exchange_current_density = result.get('exchange_current_density', 'N/A')
                overpotential_10_mA_cm2 = result.get(overpotential_key, 'N/A')
                
                # Write each row for the sample
                writer.writerow([
                    sample_label, 
                    cycle, 
                    slope, 
                    intercept, 
                    r_squared, 
                    exchange_current_density, 
                    overpotential_10_mA_cm2
                ])

    print(f"Tafel regression results saved to: {output_file}")

    
def plot_tafel_results(tafel_regression_results, folder, plot_params, base_params):
    """
    Plots the evolution of Tafel slopes, exchange current densities, and overpotential at 10 mA/cm² for each sample and cycle.

    Parameters:
        tafel_regression_results (list of dict): Regression results with slopes, exchange current densities, and overpotential.
        folder (str): Path to save the plots (e.g., 'path_OUT\\name\\Tafel').
        plot_params (dict): Plot customization parameters.
    """
    overpotential_limit = base_params.get('overpotential', 10)
    overpotential_label = f'Overpotential at {overpotential_limit} mA/cm² (mV)'
    overpotential_key = f"overpotential_{overpotential_limit}_mA_cm2"
    # Initialize plots for Tafel slopes, exchange current densities, and overpotential
    fig_slope, ax_slope = plt.subplots()
    fig_i0, ax_i0 = plt.subplots()
    fig_overpotential, ax_overpotential = plt.subplots()
    
    for sample_idx, sample in enumerate(tafel_regression_results):
        sample_label = sample['sample_label']
        results = sample['results']

        # Extract cycles, slopes, exchange current densities, and overpotential values
        cycles = [result['cycle'] for result in results if result['slope'] is not None]
        slopes = [result['slope'] for result in results if result['slope'] is not None]
        i0_values = [result['exchange_current_density'] for result in results if result['exchange_current_density'] is not None]
        overpotential_values = [result[overpotential_key] for result in results if result[overpotential_key] is not None]

        # Generate unique colors for each sample
        color = plt.cm.tab10(sample_idx % 10)

        # Plot Tafel slopes
        ax_slope.plot(cycles, slopes, label=sample_label, marker='o', color=color, linewidth=plot_params.get('linewidth', 1.5))
        
        # Plot exchange current densities
        ax_i0.plot(cycles, i0_values, label=sample_label, marker='o', color=color, linewidth=plot_params.get('linewidth', 1.5))
        
        # Plot overpotential at 10 mA/cm²
        ax_overpotential.plot(cycles, overpotential_values, label=sample_label, marker='o', color=color, linewidth=plot_params.get('linewidth', 1.5))

    # Customize Tafel slope plot
    ax_slope.set_xlabel('Cycle Number', fontsize=plot_params.get('axis_label_fontsize', 14))
    ax_slope.set_ylabel('Tafel Slope (mV/dec)', fontsize=plot_params.get('axis_label_fontsize', 14))
    ax_slope.legend(fontsize=plot_params.get('legend_fontsize', 12), frameon=False)
    ax_slope.tick_params(labelsize=plot_params.get('tick_fontsize', 12))

    # Customize exchange current density plot
    ax_i0.set_xlabel('Cycle Number', fontsize=plot_params.get('axis_label_fontsize', 14))
    ax_i0.set_ylabel(r'Exchange Current Density ($A/cm^2$)', fontsize=plot_params.get('axis_label_fontsize', 14))
    ax_i0.legend(fontsize=plot_params.get('legend_fontsize', 12), frameon=False)
    ax_i0.tick_params(labelsize=plot_params.get('tick_fontsize', 12))

    # Customize overpotential plot
    ax_overpotential.set_xlabel('Cycle Number', fontsize=plot_params.get('axis_label_fontsize', 14))
    ax_overpotential.set_ylabel(overpotential_label, fontsize=plot_params.get('axis_label_fontsize', 14))
    ax_overpotential.legend(fontsize=plot_params.get('legend_fontsize', 12), frameon=False)
    ax_overpotential.tick_params(labelsize=plot_params.get('tick_fontsize', 12))

    # Save plots
    slope_plot_path = f"{folder}_slope_evolution.png"
    i0_plot_path = f"{folder}_i0_evolution.png"
    overpotential_plot_path = f"{folder}_overpotential_evolution.png"
    fig_slope.savefig(slope_plot_path, bbox_inches='tight', dpi=300)
    fig_i0.savefig(i0_plot_path, bbox_inches='tight', dpi=300)
    fig_overpotential.savefig(overpotential_plot_path, bbox_inches='tight', dpi=300)

    plt.show()


    
from impedance.models.circuits import CustomCircuit
import numpy as np

def EIS_fit(corrected_data, EIS_E, circ='R_0-p(R_1,CPE_1)', 
            initial=[0.01220557, 0.06365453, 0.31149243, 0.62819865], 
            bound=([0.01, 0, 0, 0.6], [np.inf, np.inf, np.inf, 1])):
    
    llengthIS = 51  # Typical length of an EIS measurement segment

    fitted = []
    parameters = []
    param_names = []
    circuits = []
    
    for i, dataset_group in enumerate(corrected_data):  
        if len(dataset_group) == 0:
            continue  # Skip empty datasets

        for dataset in dataset_group:
            if len(dataset) == 0:
                continue  # Skip empty curves

            # **Extract unique voltages (rounded to 3 decimals)**
            unique_voltages = np.unique(np.round(dataset[:, 6], 3))
            selected_voltages = [v for v in unique_voltages if v in EIS_E]  # ✅ Only keep requested voltages

            print(f"Dataset {i}: Unique voltages selected: {selected_voltages}")

            for voltage in selected_voltages:
                # **Select only rows matching this voltage**
                voltage_filtered_data = dataset[np.round(dataset[:, 6], 3) == voltage]

                if len(voltage_filtered_data) == 0:
                    continue  # Skip empty selections

                # **Split into loops (3 measurements per PEIS)**
                if len(voltage_filtered_data) >= 3 * llengthIS * 0.9:
                    dataL = np.array_split(voltage_filtered_data, 3)
                elif (len(voltage_filtered_data) < 3 * llengthIS * 0.9) and (len(voltage_filtered_data) >= 1 * llengthIS * 1.1):
                    dataL = np.array_split(voltage_filtered_data, 2)
                else:
                    dataL = [voltage_filtered_data]

                for data in dataL:
                    frequencies = data[:, 0]  
                    Z = np.vectorize(complex)(data[:, 1], -1 * data[:, 2])  

                    # 🛑 Remove NaN and Inf values
                    valid_indices = np.isfinite(frequencies) & np.isfinite(Z.real) & np.isfinite(Z.imag)
                    frequencies = frequencies[valid_indices]
                    Z = Z[valid_indices]

                    # 🚨 Skip dataset if empty after filtering
                    if len(frequencies) == 0:
                        print(f"Skipping dataset {i} at voltage {voltage} due to all NaN/Inf values")
                        continue  

                    # **Initialize and Fit Circuit**
                    circuit = CustomCircuit(initial_guess=initial, circuit=circ)
                    try:
                        circuit.fit(frequencies, Z, bounds=bound)
                    except Exception as e:
                        print(f"Error during fitting dataset {i} at voltage {voltage}: {e}")
                        continue  

                    # **Store results**
                    fitted_real = circuit.predict(frequencies).real
                    fitted_imag = -circuit.predict(frequencies).imag
                    
                    fitted.append([fitted_real, fitted_imag])
                    parameters.append(vars(circuit)['parameters_'])
                    param_names.append(circuit.get_param_names())
                    circuits.append(circuit)

    return fitted, parameters, param_names, circuits



def save_eis_results_to_csv(eis_fitting_results, user_params_list, output_file, EIS_E):
    """
    Saves EIS fitting results to a CSV file with sample metadata and fit errors.

    Parameters:
        eis_fitting_results (tuple): Output from `EIS_fit` function containing:
            - fitted_data (list): Real and imaginary components of fitted impedance.
            - parameters (list of np.array): Fitted circuit parameters for each dataset.
            - param_names (list of tuples): Contains (parameter names, units).
            - circuits (list): Circuit models used.
        user_params_list (list of dict): Contains sample metadata (label, day, loop, etc.).
        output_file (str): Path to the output CSV file.
        EIS_E (list): List of voltages used for EIS fitting.
    """

    fitted_data, parameters, param_names, circuits = eis_fitting_results

    # Ensure there is at least one dataset
    if not parameters:
        print("No valid EIS fitting results to save.")
        return

    # Extract parameter names and units from the first dataset
    param_labels, param_units = param_names[0]  # Get names and units from first dataset

    # Metadata column names (assumed same for all user_param_list entries)
    metadata_columns = ["Label", "Day", "Loop", "Proc_Nr", "Seq", "S (cm²)", "R_ohm (Ohm)", "E (V)"]

    # Create column headers with units and errors
    param_columns = [f"{name} ({unit})" for name, unit in zip(param_labels, param_units)]
    error_columns = [f"Error {name} ({unit})" for name, unit in zip(param_labels, param_units)]
    header = metadata_columns + ["Circuit Model"] + param_columns + error_columns  # ✅ Add error columns

    # Write to CSV
    with open(output_file, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        
        # Write header row
        writer.writerow(header)

        # Write data rows
        index = 0  # Keep track of which sample corresponds to which fit
        for i, sample_metadata in enumerate(user_params_list):
            sample_label = sample_metadata.get("label", f"Sample_{i+1}")  # Default name if missing
            day = sample_metadata.get("day", ["N/A"])[0]  # Extract first day entry
            loop = sample_metadata.get("loop", ["N/A"])[0]  # Extract first loop entry
            proc_nr = sample_metadata.get("proc_nr", ["N/A"])[0]  # Extract first process number
            seq = sample_metadata.get("seq", "N/A")  # Sequence
            S = sample_metadata.get("S", "N/A")  # Sample area
            R_ohm = sample_metadata.get("R_ohm", ["N/A"])[0]  # Extract first R_ohm entry

            for voltage in EIS_E:  # Loop through each voltage used for fitting
                if index >= len(parameters):  # Ensure we don't exceed available fits
                    break
                
                circuit_model = str(circuits[index].circuit)  # Ensure circuit model is a string
                param_values_list = parameters[index].tolist() if isinstance(parameters[index], np.ndarray) else parameters[index]

                # ✅ Get fit errors
                try:
                    fit_errors = circuits[index].conf_.tolist() if circuits[index].conf_ is not None else ["N/A"] * len(param_values_list)
                except AttributeError:
                    fit_errors = ["N/A"] * len(param_values_list)  # In case fit errors are not available

                # Write row: Metadata + Circuit Model + Fitted Parameter Values + Fit Errors
                writer.writerow([sample_label, day, loop, proc_nr, seq, S, R_ohm, voltage, circuit_model] + param_values_list + fit_errors)
                
                index += 1  # Move to next fit

    print(f"EIS fitting results saved to: {output_file}")
