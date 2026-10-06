"""Diagnostic: observed sampling gaps, repeated PRBS folding, and spectral ratio."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from calculate_prbs_impedance import read_mpr, get_ewe_column, load_geis, DATA_ROOT


def folding_diagnostic(d, lo, hi, fmax):
    t = d['time/s'].to_numpy(float)
    ns = d.Ns.to_numpy(int)
    dt = np.median(np.diff(t))
    # True step starts must have a short preceding interval. The first point
    # after a communication gap is an interval average and cannot be used.
    gap = np.r_[True, np.diff(t) > 2.1 * dt]
    transition = np.r_[False, np.diff(ns) != 0]
    mask = (ns >= lo) & (ns <= hi)
    reset = np.r_[False, np.diff(ns) < 0] & mask
    cyc = np.cumsum(reset)
    cyc -= cyc[np.flatnonzero(mask)[0]]
    starts = pd.DataFrame({'t': t, 'ns': ns, 'cycle': cyc})[mask & transition & ~gap]
    periods = []
    for _, g in starts.groupby('ns'):
        a, b = g.t.to_numpy(), g.cycle.to_numpy()
        periods.extend((np.diff(a) / np.diff(b)).tolist())
    period = float(np.median(periods))
    clean_first = starts[starts.ns == lo]
    epoch = float(np.median(clean_first.t - clean_first.cycle * period))
    n = int(round(period / dt))
    print('band', lo, hi, 'dt', dt, 'period', period, 'epoch', epoch, 'cycles', cyc[mask].max()+1, 'phase bins', n, flush=True)
    phase = np.rint((t - epoch - cyc * period)/dt).astype(int) % n
    phase_error = (t-epoch-cyc*period) / dt - np.rint((t-epoch-cyc*period)/dt)
    print('lattice residual p99', np.quantile(abs(phase_error[mask]), .99), flush=True)
    e = d[get_ewe_column(d,'PRBS')].to_numpy(float)
    i = d['I/mA'].to_numpy(float)/1000
    all_z = []
    for mode in ['interpolate', 'fold_all', 'fold_clean']:
        if mode == 'interpolate':
            tt = np.arange(t[mask][0],t[mask][-1],dt)
            ii, ee = np.interp(tt,t[mask],i[mask]), np.interp(tt,t[mask],e[mask])
            kw=dict(fs=1/dt,nperseg=n,noverlap=0,detrend='linear')
            f,p = signal.welch(ii,**kw); _,c=signal.csd(ii,ee,**kw)
            z=c/p
        else:
            ok = mask & (cyc >= 1)
            if mode=='fold_clean':
                ok &= ~gap
            counts=np.bincount(phase[ok], minlength=n)
            print(mode,'coverage',np.mean(counts>0), 'min',counts.min(), 'median',np.median(counts),flush=True)
            ii=np.bincount(phase[ok],weights=i[ok],minlength=n)/np.maximum(counts,1)
            # Subtract each cycle mean before folding to suppress DC drift.
            ee0=e.copy()
            for cycle in np.unique(cyc[ok]):
                sel=ok & (cyc==cycle)
                ee0[sel]-=np.mean(ee0[sel])
            ee=np.bincount(phase[ok],weights=ee0[ok],minlength=n)/np.maximum(counts,1)
            grid=np.arange(n)
            ii=np.interp(grid,grid[counts>0],ii[counts>0],period=n)
            ee=np.interp(grid,grid[counts>0],ee[counts>0],period=n)
            f=np.fft.rfftfreq(n,dt)
            fi=np.fft.rfft(ii); fe=np.fft.rfft(ee)
            z=np.divide(fe,fi,out=np.full_like(fe,np.nan),where=abs(fi)>1e-15)
            z[abs(fi)<0.02*np.max(abs(fi[1:]))]=np.nan
        select=(f>=2)&(f<=fmax)
        all_z.append(pd.DataFrame({'frequency_Hz':f[select], 'Re_Z_ohm':z.real[select], 'Im_Z_ohm':z.imag[select], 'mode':mode}))
    return pd.concat(all_z)


def main():
    d=read_mpr(DATA_ROOT/'VIII_Day9_PRBS_500mA_20mA_amp_C01.mpr')
    g=load_geis(read_mpr(DATA_ROOT/'VIII_Day9_GEIS_500mA_C01.mpr'))
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for col,(lo,hi,fmax) in enumerate([(1,196,30),(197,708,80)]):
        r=folding_diagnostic(d,lo,hi,fmax)
        for mode,q in r.groupby('mode'):
            f=q.frequency_Hz.to_numpy(); z=q.Re_Z_ohm.to_numpy()+1j*q.Im_Z_ohm.to_numpy()
            zg=np.interp(np.log(f),np.log(g['freq/Hz']),g['Re(Z)/Ohm'])+1j*np.interp(np.log(f),np.log(g['freq/Hz']),g.Im_Z_ohm)
            print(lo,mode,'n',np.isfinite(z).sum(),'median complex error %',np.nanmedian(abs(z-zg)/abs(zg))*100,'phase median',np.nanmedian(np.angle(z/zg,deg=True)),flush=True)
            axes[0,col].semilogx(f,abs(z),'.',ms=2,label=mode)
            axes[1,col].semilogx(f,np.angle(z,deg=True),'.',ms=2,label=mode)
        sel=g['freq/Hz'].between(2,fmax)
        axes[0,col].semilogx(g.loc[sel,'freq/Hz'],g.loc[sel,'abs_Z_ohm'],'k.-',label='GEIS')
        axes[1,col].semilogx(g.loc[sel,'freq/Hz'],g.loc[sel,'phase_deg'],'k.-',label='GEIS')
        axes[0,col].set_ylim(0,.1); axes[1,col].set_ylim(-60,20)
        axes[0,col].legend(); axes[1,col].set_xlabel('Frequency / Hz')
        axes[0,col].set_ylabel('|Z| / ohm'); axes[1,col].set_ylabel('phase / deg')
    fig.savefig('prbs_folding_diagnostic.png',dpi=150)


if __name__=='__main__':
    main()
