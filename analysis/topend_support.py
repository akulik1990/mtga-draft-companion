"""Pre-registered: does top end need support (early time-buyers, mana)? Writes docs/topend-support-FRA.md."""
import gzip, json, re, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
R = '/home/morris/Projects/mtga-draft-companion/'
cards = json.load(open(R + 'data/Sets/FRA_PremierDraft_All_Data.json'))['card_ratings']
decks = json.load(gzip.open(R + 'analysis/out/decks_FRA.json.gz'))
BASIC = {'Plains', 'Island', 'Swamp', 'Mountain', 'Forest', 'Wastes'}
TEMPO = re.compile(r"counter target|return target [^.]*to (its|their) owner's hand|tap target|stun counter|doesn't untap", re.I)
info = {}
for c in cards.values():
    n = c['name'].split(' // ')[0]
    if n in info and info[n]['cmc'] > 0 and 'Land' in c['types']: continue
    t = c.get('oracle_text', '') or ''; tl = t.lower(); land = 'Land' in c['types']; kw = c.get('keywords', [])
    cmc = c['cmc']; creature = 'Creature' in c['types']
    info[n] = dict(land=land, cmc=cmc, dc=c['deck_colors'],
        removal='removal' in c.get('tags', []),
        tempo=(not creature) and cmc <= 2 and bool(TEMPO.search(t)),
        # toughness is not in the data: defence = Defender only
        defence=creature and ('Defender' in kw or 'defender' in tl.split('\n')[0].lower().split(',')[0] or bool(re.search(r'\bdefender\b', tl))),
        lcyc='landcycling' in tl or any('andcycling' in k for k in kw),
        ramp_old=(not land) and ('add {' in tl or 'search your library for a basic land' in tl),
        # amendment 1: + cost reducers + treasure makers, cmc >= 7 excluded
        ramp=(not land) and cmc < 7 and ('add {' in tl or 'search your library for a basic land' in tl or bool(re.search(r"costs? \{\d\} less", tl)) or 'treasure' in tl or 'Treasure' in kw),
        # amendment 2: land colours (non-basic, from oracle) and draw/selection
        lcol=set(re.findall(r'\{([WUBRG])\}', ' '.join(re.findall(r'add ([^.\n]*)', t, re.I)))) if land else set(),
        anycol=land and bool(re.search(r'any color|chosen color|color of your choice|any type', tl)),
        draw=(not land) and ('card_advantage' in c.get('tags', []) or bool(re.search(r'draw (a|two|three) cards?|surveil \d|scry \d', tl))))
def gih(i, lane):
    for k in (lane, lane[::-1]):
        v = i['dc'].get(k, {}).get('gihwr')
        if v: return v
    return i['dc'].get('All Decks', {}).get('gihwr')
rows = []
for d in decks:
    if d['skill'] < .60 or len(d['lane']) != 2 or d['deck_games'] < 1 or sum(d['counts'].values()) != 40: continue
    r = dict(draft=d['draft'], lane=d['lane'], skill=d['skill'], gb=d['games_bucket'], w=d['deck_wins'], l=d['deck_games'] - d['deck_wins'],
             top=0, top6=0, rem=0, tem=0, dfn=0, mana=0, mana_old=0, duals=0, draw=0, q=0., ns=0)
    for n, k in d['counts'].items():
        i = info.get(n)
        if not i: continue
        if i['land']:
            r['mana'] += k; r['mana_old'] += k
            if n not in BASIC and (len(i['lcol'] & set(d['lane'])) >= 2 or i['anycol']): r['duals'] += k
            continue
        r['ns'] += k; g = gih(i, d['lane']); r['q'] += k * g if g else 0
        if i['cmc'] >= 5: r['top'] += k
        if i['cmc'] >= 6: r['top6'] += k
        if i['lcyc'] or i['ramp']: r['mana'] += k
        if i['lcyc'] or i['ramp_old']: r['mana_old'] += k
        r['draw'] += k * i['draw']
        if i['cmc'] <= 3:
            r['rem'] += k * i['removal']; r['tem'] += k * i['tempo']; r['dfn'] += k * i['defence']
    r['q'] /= max(r['ns'], 1); rows.append(r)
df = pd.DataFrame(rows)
df['early'] = df.rem + df.tem + df.dfn  # note: a card in two subgroups counts twice in the subgroup split
df['mh'] = (df.mana >= 18).astype(int); df['mh_old'] = (df.mana_old >= 18).astype(int); df['gbc'] = df.gb.astype(str)
sub = {k: sum(1 for i in info.values() if not i['land'] and i['cmc'] <= 3 and i[k]) for k in ('removal', 'tempo', 'defence')}
k = .55 * .45 * 100  # logit coef -> win-rate points at 55%
ctrl = '+ q + skill + C(gbc) + C(lane)'
def fit(f, d=df):
    return smf.glm(f, d, family=sm.families.Binomial(), freq_weights=None).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.draft)[0]})
def mk(top, mh='mh'):
    return smf.glm(f'I(w/(w+l)) ~ {top}*early + {top}*{mh} + q + skill + C(gbc) + C(lane)', df, family=sm.families.Binomial(), var_weights=df.w + df.l)
def run(top, mh='mh'):
    m = mk(top, mh); return m.fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(df.draft)[0]})
def tab(res, names):
    out = ['| term | coef (logit) | win-rate pts at 55% | 95% CI pts | p |', '|---|---|---|---|---|']
    ci = res.conf_int()
    for n in names:
        out.append(f'| {n} | {res.params[n]:.4f} | {res.params[n]*k:+.3f} | {ci.loc[n,0]*k:+.3f} to {ci.loc[n,1]*k:+.3f} | {res.pvalues[n]:.3f} |')
    return '\n'.join(out)
terms = lambda top: [top, 'early', 'mh', f'{top}:early', f'{top}:mh']
def ratio(res, top, mh):
    # early* = -(b_top + b_tm*mh)/b_te ; delta method on clustered cov
    p = res.params; V = res.cov_params(); a, b, c = f'{top}', f'{top}:early', f'{top}:mh'
    num = p[a] + p[c] * mh; e = -num / p[b]
    g = pd.Series(0., index=p.index); g[a] = -1 / p[b]; g[c] = -mh / p[b]; g[b] = num / p[b] ** 2
    se = float(np.sqrt(g @ V @ g)); return e, se
doc = []
res = {}
for top in ('top', 'top6'):
    res[top] = run(top)
# subgroup model (top >=5 only, and top6)
def sg(top):
    f = f'I(w/(w+l)) ~ {top}*(rem + tem + dfn) + {top}*mh + q + skill + C(gbc) + C(lane)'
    m = smf.glm(f, df, family=sm.families.Binomial(), var_weights=df.w + df.l)
    return m.fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(df.draft)[0]})
S = {t: sg(t) for t in ('top', 'top6')}
n, nd = len(df), df.draft.nunique()
desc = df[['top', 'top6', 'early', 'rem', 'tem', 'dfn', 'mana', 'mh', 'q']].agg(['mean', 'std']).round(2).T
g = df.w.sum() + df.l.sum()
L = [f'# Top-end support, FRA (pre-registered analysis)', '',
 'Script: `analysis/topend_support.py` (rerun regenerates this file).', '',
 '## Question and rule', '',
 'Hypothesis: expensive cards pay off only if the deck can reach the mana and survive to cast them, and survival tools (early removal, tempo, defence) are interchangeable. '
 'Rule fixed before the run: if top_end x early is significantly positive (p < 0.05), the builder should require early time-buyers in proportion to top end, reported as the number of early cards at which the per-card top-end effect reaches 0 (delta-method CI, clustered covariance); same for top_end x mana_high. Otherwise the verdict is "no evidence that top end needs support beyond card quality". Subgroups substitute if their interaction CIs overlap.', '',
 '## Definitions', '',
 'top_end = spells with cmc >= 5 (copies); top6 = cmc >= 6. early = per copy, cmc <= 3 and any of: tagged removal; tempo (non-creature, cmc <= 2, oracle matches counter / bounce / tap / stun / doesn\'t untap); defence (creature with Defender in oracle). Toughness is not in the data, so the toughness >= 4 clause was skipped. '
 f'Distinct cards per subgroup in the set: removal {sub["removal"]}, tempo {sub["tempo"]}, defence {sub["defence"]}. A card matching two subgroups counts twice in the split model. '
 'mana = lands + landcycling cards + ramp spells. Ramp (amended definition, primary): non-land, cmc < 7, oracle has "add {", "search your library for a basic land", a cost reducer ("cost(s) {n} less") or "treasure". Old definition (secondary line below): no cost reducers or treasure makers, no cmc cap. duals = non-basic lands whose "Add" text gives 2+ colours of the lane, or that tap for a chosen or any colour (copies). draw = non-land spells tagged card_advantage or with oracle matching draw a/two/three card(s), surveil N or scry N (copies). mana_high = mana >= 18. quality = copy-weighted mean lane GIH WR of spells (fallback All Decks). Oracle text exists for only a subset of cards in the file, so tempo/ramp/defence detection is a floor.', '',
 '## Sample', '',
 f'skill >= 0.60, 40-card decks, two-colour lanes, deck_games >= 1: {n} decks from {nd} drafts, {g:.0f} games; mean deck win rate {df.w.sum()/g:.3f}. Splash decks are included (lane is the two-colour label).', '',
 desc.to_markdown(), '',
 'Model: binomial GLM on wins/games (variance weights = games), controls quality, skill, games bucket, lane; SEs clustered by draft. Win-rate points = logit coefficient x 0.55 x 0.45 x 100.', '']
for top, lab in (('top', 'cmc >= 5'), ('top6', 'cmc >= 6')):
    r = res[top]
    L += [f'## Main model, top end {lab}', '', tab(r, terms(top)), '']
    pe = r.pvalues[f'{top}:early']; pm = r.pvalues[f'{top}:mh']
    for mh in (0, 1):
        e, se = ratio(r, top, mh)
        L.append(f'Per-card top-end effect is zero at early = {e:.1f} (95% CI {e-1.96*se:.1f} to {e+1.96*se:.1f}) when mana_high = {mh}; sample mean early {df.early.mean():.1f}, top end {df[top].mean():.1f}.')
    L += ['', f'top x early: coef {r.params[f"{top}:early"]*k:+.3f} pts, p = {pe:.3f}. top x mana_high: {r.params[f"{top}:mh"]*k:+.3f} pts, p = {pm:.3f}.', '']
for top in ('top', 'top6'):
    r = S[top]; names = [top, 'rem', 'tem', 'dfn', 'mh', f'{top}:rem', f'{top}:tem', f'{top}:dfn', f'{top}:mh']
    ci = r.conf_int()
    L += [f'## Early split into subgroups ({top})', '', tab(r, names), '']
    ov = []
    for a, b in (('rem', 'tem'), ('rem', 'dfn'), ('tem', 'dfn')):
        A, B = f'{top}:{a}', f'{top}:{b}'
        ov.append(f'{a}/{b}: ' + ('CIs overlap' if max(ci.loc[A, 0], ci.loc[B, 0]) <= min(ci.loc[A, 1], ci.loc[B, 1]) else 'CIs disjoint'))
    L += ['Substitution check: ' + '; '.join(ov) + '. (Overlap of wide CIs is weak evidence of equality; tempo and defence are rare.)', '']
r = res['top']; pe = r.pvalues['top:early']; be = r.params['top:early']; pm = r.pvalues['top:mh']; bm = r.params['top:mh']
sig_e = pe < .05 and be > 0; sig_m = pm < .05 and bm > 0
e0, se0 = ratio(r, 'top', 0)
verdict = []
verdict.append(('Early: significantly positive interaction; builder should require early time-buyers, break-even early count ' + f'{e0:.1f} (CI {e0-1.96*se0:.1f} to {e0+1.96*se0:.1f}) at mana_high = 0.') if sig_e else 'Early: no evidence that top end needs support beyond card quality (top x early not significantly positive).')
verdict.append('Mana: top x mana_high significantly positive; builder should require mana >= 18 for top end.' if sig_m else 'Mana: no evidence that top end needs mana support beyond card quality (top x mana_high not significantly positive).')
ro = run('top', 'mh_old'); ci = ro.conf_int()
L += ['## Secondary: old mana definition', '', f'Old mana_high ({df.mh_old.mean():.2f} of decks vs {df.mh.mean():.2f} under the amended one): top x mana_high {ro.params["top:mh_old"]*k:+.3f} pts (95% CI {ci.loc["top:mh_old",0]*k:+.3f} to {ci.loc["top:mh_old",1]*k:+.3f}), p = {ro.pvalues["top:mh_old"]:.3f}; top x early {ro.params["top:early"]*k:+.3f} pts, p = {ro.pvalues["top:early"]:.3f}.', '']
rl = sorted(n for n, i in info.items() if (i['ramp'] or i['lcyc']) and not i['land'])
L += ['## Ramp list (amended definition, incl. landcyclers)', '', '; '.join(f'{n} ({info[n]["cmc"]})' for n in rl), '']
JT = {}
for top in ('top', 'top6'):
    f = f'I(w/(w+l)) ~ {top}*(early + mh + duals + draw) + q + skill + C(gbc) + C(lane)'
    JT[top] = smf.glm(f, df, family=sm.families.Binomial(), var_weights=df.w + df.l).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(df.draft)[0]})
for top in ('top', 'top6'):
    r = JT[top]; ci = r.conf_int(); sup = ['early', 'mh', 'duals', 'draw']
    L += [f'## Joint support model ({top}): early, mana_high, duals, draw each interacted with top end', '',
          f'Means: duals {df.duals.mean():.2f} (sd {df.duals.std():.2f}), draw {df.draw.mean():.2f} (sd {df.draw.std():.2f}). Per-unit effects (one more copy; mana_high is a 0/1 switch).', '',
          tab(r, [top] + sup + [f'{top}:{x}' for x in sup]), '']
    ov = []
    for a in range(4):
        for b in range(a + 1, 4):
            A, B = f'{top}:{sup[a]}', f'{top}:{sup[b]}'
            ov.append(f'{sup[a]}/{sup[b]}: ' + ('overlap' if max(ci.loc[A, 0], ci.loc[B, 0]) <= min(ci.loc[A, 1], ci.loc[B, 1]) else 'disjoint'))
    L += ['Interaction CI overlap (units differ, so overlap is only a loose substitution check): ' + '; '.join(ov) + '.', '']
jr = JT['top']
verdict.append('Joint model, separate interactions (top end cmc >= 5): ' + '; '.join(f'top x {x} {jr.params["top:"+x]*k:+.3f} pts, p = {jr.pvalues["top:"+x]:.3f}' for x in ('early', 'mh', 'duals', 'draw')) + '. Significantly positive: ' + (', '.join(x for x in ('early', 'mh', 'duals', 'draw') if jr.pvalues['top:'+x] < .05 and jr.params['top:'+x] > 0) or 'none') + '.')
L += ['## Verdict', ''] + verdict + ['',
 '## Caveats', '',
 'Selection: these are the 60%+ players\' decks, not randomised. Players with a heavy top end who build it anyway also pick their support differently, and control players build differently from aggro players; deck archetype is not controlled beyond lane. Card quality is controlled only through 17Lands GIH WR averages, which already include each card\'s typical support, so some of the support effect is absorbed into quality and the interaction is biased toward zero. The defence subgroup is a single card (one Defender creature in the set), so its coefficients, including the cmc >= 6 result at p = 0.01 that is not corrected for multiplicity, describe that one card and not a defence class. The break-even early counts are meaningless when the interaction is not significant (CIs span zero and negative values). Binomial weights are games per deck via variance weights, and statsmodels warns that clustered SEs are not fully supported with them. Early counts are a floor because oracle text is missing for many cards, and toughness-based defence is absent. Two main tests plus two for cmc >= 6 and the subgroup models are not corrected for multiplicity; treat p near 0.05 as weak. Win rate is over few games per deck (see games bucket), so deck-level noise is large; clustering is by draft, not by player.']
open(R + 'docs/topend-support-FRA.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
