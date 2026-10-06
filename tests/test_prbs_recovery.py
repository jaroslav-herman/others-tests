"""Independent sign, recovery, refusal and sequence-period regression tests."""
import unittest
import numpy as np
import pandas as pd
from scipy.signal import max_len_seq

from calculate_prbs_impedance import morlet_coefficients, interpolate_uniform, read_mpr, DATA_ROOT
from generate_prbs_cp_mps import maximal_length_prbs, technique_from_runs
from prbs_repeated_spectrum import estimate_repeated_band, periodic_fill


class PRBSRecoveryTests(unittest.TestCase):
    def test_cwt_positive_and_negative_phase(self):
        dt=.0002
        t=np.arange(0, 2, dt)
        for phase in (-.5,.5):
            i=np.cos(2*np.pi*40*t)
            e=.07*np.cos(2*np.pi*40*t+phase)
            ci,_,_=morlet_coefficients(i,dt,np.array([40.]))
            ce,_,_=morlet_coefficients(e,dt,np.array([40.]))
            sel=(t>.4)&(t<1.6)
            measured=np.mean(ce[0,sel]/ci[0,sel])
            self.assertLess(abs(measured-.07*np.exp(1j*phase)),1e-6)

    def test_all_mls_states_and_circular_autocorrelation(self):
        for order in (8,9,10,11,14,15,16):
            bits=np.array(maximal_length_prbs(order))
            self.assertEqual(len(bits),2**order-1)
            self.assertEqual(bits.sum(),2**(order-1))
            x=2*bits-1
            ac=np.fft.ifft(abs(np.fft.fft(x))**2).real
            self.assertLess(np.max(abs(ac[1:]+1)),1e-7)

    def test_large_gaps_are_not_silently_interpolated(self):
        t=np.r_[np.arange(50)*.0002,.03+np.arange(50)*.0002]
        with self.assertRaisesRegex(ValueError,'timestamp gaps'):
            interpolate_uniform(t,np.sin(t))
        with self.assertRaisesRegex(ValueError,'Unobserved phase interval'):
            periodic_fill(np.ones(100),np.arange(100)<40,.0002,.001)

    def test_fixed_width_settings(self):
        content=technique_from_runs(1,[(None,30.),(1,.004),(0,.008)],500,20)
        rows={line[:20].strip():[line[j:j+20].strip() for j in range(20,len(line),20)]
              for line in content.splitlines()[2:]}
        self.assertEqual(rows['record'],['Ewe']*3)
        self.assertEqual(rows['dQM'],['0.000000']*3)
        self.assertEqual(rows['dts (s)'],['0.0010']*3)
        self.assertEqual(rows['Is'],['500.000','520.000','480.000'])
        self.assertTrue(all(len(values)==3 for values in rows.values()))

    def test_gapped_repeated_rc_and_positive_phase_network(self):
        rng=np.random.default_rng(27)
        dt=.0002
        bits=max_len_seq(7)[0]
        i=.5+.02*np.repeat(2*bits.astype(float)-1,20)
        n=len(i)
        ns=np.cumsum(np.r_[True,np.diff(i)!=0])
        frequency=np.fft.rfftfreq(n,dt)
        base=.04/(1+1j*2*np.pi*frequency*.003)
        for network in (.03+base, .07-base):
            voltage=1.5+np.fft.irfft(np.fft.rfft(i-i.mean())*network,n=n)
            copies=14
            tt=np.arange(copies*n)*dt
            phase=np.arange(copies*n)%n
            ii=np.tile(i,copies)
            ee=np.tile(voltage,copies)
            keep=np.ones(len(tt),bool)
            # Different 11 ms gaps in each period, approximately 35% missing.
            for cycle in range(copies):
                for start in rng.choice(np.arange(100,n-100),size=20,replace=False):
                    keep[cycle*n+start:cycle*n+start+55]=False
            # First value after a gap is an interval average, like recorded <Ewe>.
            ix=np.flatnonzero(keep)
            observed=ee[ix].copy()
            for j in np.flatnonzero(np.r_[False,np.diff(ix)>1]):
                observed[j]=ee[ix[j-1]+1:ix[j]+1].mean()
            data=pd.DataFrame({'time/s':tt[ix], 'Ns':ns[phase[ix]],
                               'I/mA':ii[ix]*1000,'<Ewe>/V':observed})
            r,a,_=estimate_repeated_band(data,'<Ewe>/V',int(ns.min()),int(ns.max()),2,80,'test')
            expected=np.interp(r.frequency_Hz,frequency,network.real)+1j*np.interp(r.frequency_Hz,frequency,network.imag)
            measured=r.Re_Z_ohm.to_numpy()+1j*r.Im_Z_ohm.to_numpy()
            self.assertLess(np.median(abs(measured-expected)/abs(expected)),.005)
            self.assertLess(np.quantile(abs(measured-expected)/abs(expected),.9),.02)
            self.assertGreater(a['missing_grid_fraction'],.2)

    @unittest.skipUnless((DATA_ROOT/'VIII_Day9_PRBS_500mA_20mA_amp_C01.mpr').exists(),'local MPR integration check')
    def test_real_mpr_legacy_ids_and_charge(self):
        d=read_mpr(DATA_ROOT/'VIII_Day9_PRBS_500mA_20mA_amp_C01.mpr')
        self.assertIn('<Ewe>/V',d)
        self.assertNotIn('Phase(Zwe-ce)/deg',d)
        self.assertNotIn('unknown_4bytes',d)
        np.testing.assert_allclose(d['Q charge/discharge/mA.h']*3.6,d['(Q-Qo)/C'],rtol=2e-6,atol=1e-6)


if __name__=='__main__':
    unittest.main()
