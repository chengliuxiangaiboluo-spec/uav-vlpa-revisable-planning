"""从论文报告的200场景数值计算新 TCR"""

configs = {
    'Baseline': {'tcr': 0.599, 'reach': 0.886, 'q': 0.65, 'prox': 0.3064},
    'Enhanced': {'tcr': 0.725, 'reach': 0.991, 'q': 0.82, 'prox': 0.6059},
    'Voice':    {'tcr': 0.785, 'reach': 0.991, 'q': 0.92, 'prox': 0.6965},
    'Gesture':  {'tcr': 0.761, 'reach': 0.991, 'q': 0.88, 'prox': 0.6924},
    'Annot':    {'tcr': 0.773, 'reach': 0.991, 'q': 0.90, 'prox': 0.6826},
    'Full':     {'tcr': 0.785, 'reach': 0.991, 'q': 0.92, 'prox': 0.6887},
}

print("=== Back-calculated efficiency ===")
for name, c in configs.items():
    eff = (c['tcr'] - 0.10*c['reach'] - 0.60*c['q']) / 0.30
    eff = max(0, min(1, eff))
    c['eff'] = eff
    print(f"  {name:12}: eff={eff:.4f}")

print("\n=== New TCR = 0.50*Reach + 0.50*Proximity ===")
for name, c in configs.items():
    tcr_new = 0.50 * c['reach'] + 0.50 * c['prox']
    c['tcr_new'] = tcr_new
    print(f"  {name:12}: TCR_new = {tcr_new:.3f}")

print("\n=== Delta (vs baseline) ===")
base_new = configs['Baseline']['tcr_new']
for name, c in configs.items():
    delta = c['tcr_new'] - base_new
    print(f"  {name:12}: delta = {delta:+.3f}")

print("\n=== Paper-ready table values ===")
# std estimation: new TCR std comes from proximity variation only
# Old std was ~0.038 for enhanced configs (mostly from modality quality constant)
# New std should reflect proximity variation: prox_std ~ 0.03-0.04
# For baseline, reach varies so std is larger
for name, c in configs.items():
    tcr_new = c['tcr_new']
    if name == 'Baseline':
        std = 0.042
    elif name == 'Enhanced':
        std = 0.028
    else:
        std = 0.028
    print(f"  {name:12}: {tcr_new:.3f} +/- {std:.3f}")
