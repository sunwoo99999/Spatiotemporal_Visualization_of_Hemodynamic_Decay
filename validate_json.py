import json, numpy as np
with open('c:/1_research/19_siggraph/proj/dataset/hrf_master_data.json', encoding='utf-8') as f:
    d = json.load(f)

meta = d['metadata']
subs = d['subjects']
print('=== METADATA ===')
print('  generated_at:', meta['generated_at'])
print('  n_subjects:', meta['n_subjects'], '| Young:', meta['n_young'], '| T2DM_Old:', meta['n_t2dm_old'])
print('  n_flagged:', meta['n_flagged'])

print()
print('=== SAMPLE: SUB701 V1 ===')
sub701 = next(s for s in subs if s['id']=='701')
v1 = sub701['V1']
for k in ['p1','q1','p2','q2','a1','a2','c','t_peak_s','peak_amp','fwhm_s','trough_amp','t_trough_s','fit_success']:
    print('  {:12s}: {}'.format(k, v1[k]))

print()
print('=== t_peak range validation (all subjects, V1) ===')
t_peaks = [s['V1']['t_peak_s'] for s in subs if s['V1']['t_peak_s'] is not None]
print('  min={:.2f}s  max={:.2f}s  mean={:.2f}s'.format(min(t_peaks), max(t_peaks), np.mean(t_peaks)))
print('  0<t_peak<10s:', all(0 < v < 10 for v in t_peaks), '({}/{})'.format(len(t_peaks), len(subs)))

print()
print('=== Group t_peak V1 mean ===')
young = [s['V1']['t_peak_s'] for s in subs if s['group']=='Young' and s['V1']['t_peak_s']]
other = [s['V1']['t_peak_s'] for s in subs if s['group']=='T2DM_Old' and s['V1']['t_peak_s']]
print('  Young:    {:.3f} +/- {:.3f} s (n={})'.format(np.mean(young), np.std(young), len(young)))
print('  T2DM_Old: {:.3f} +/- {:.3f} s (n={})'.format(np.mean(other), np.std(other), len(other)))

print()
print('=== fit_success ===')
for roi in ['V1','V2','V3']:
    ok = sum(1 for s in subs if s[roi]['fit_success'])
    print('  {}: {}/33'.format(roi, ok))
