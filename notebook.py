#!/usr/bin/env python3
"""SupeRR Selector S2: pressure-feasibility diagnostic (not final rankings).

Requirements: python3 -m pip install pandas numpy
Run: python3 pressure_feasibility.py "/path/to/the/organizer/data/folder"

Reads the six CSV files only; writes reports and diagnostic CSVs to
<data folder>/pressure_feasibility_output unless --output is supplied.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

DATA_DIR = "/Users/vansh/Downloads/SupeRR Selector Season 2 | Problem Statement/SupeRR Selector Season 2 | Data Pack"

NAMES = {
    'ball': 'ball_by_ball.csv', 'matches': 'matches.csv',
    'comps': 'competitions.csv', 'mapping': 'competition_team_player_mapping.csv',
    'xi': 'playing_eleven.csv', 'teams': 'teams.csv'
}

def locate(root, name):
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f'Missing {name} under {root}')
    if len(hits) > 1:
        print(f'WARNING: multiple {name} files; using {hits[0]}')
    return hits[0]

def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')

def num(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c].replace('', np.nan), errors='coerce')

def boolcol(s):
    return s.astype(str).str.strip().str.lower().isin(['true','1','yes','y'])

def counts(s, n=30):
    return [{'value': str(k), 'count': int(v)} for k,v in s.fillna('<BLANK>').astype(str).value_counts().head(n).items()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('data_dir', nargs='?', default=DATA_DIR, help='Path to dataset directory')
    ap.add_argument('--output', default=None)
    a=ap.parse_args()
    root=Path(a.data_dir).expanduser().resolve()
    if not root.is_dir(): sys.exit(f'Not a directory: {root}')
    out=Path(a.output).expanduser().resolve() if a.output else root/'pressure_feasibility_output'
    out.mkdir(parents=True, exist_ok=True)
    paths={k:locate(root,v) for k,v in NAMES.items()}
    print('Reading six organizer CSVs...')
    bb, matches, comps, mapping, xi, teams = [read(paths[k]) for k in ['ball','matches','comps','mapping','xi','teams']]

    must={
      'ball': ['match_id','innings','overid','over_ball_num','innings_ball_num','striker_id','bowler_id','runs','is_wicket','extras_type','extras1_runs','extras_2_runs'],
      'matches': ['match_id','comp_id','match_date','revised_target','first_inn_revised_overs','second_inn_revised_overs'],
      'comps': ['comp_id','comp_name','match_format','season']}
    tables={'ball':bb,'matches':matches,'comps':comps}
    missing={k:[c for c in cs if c not in tables[k].columns] for k,cs in must.items()}
    if any(missing.values()):
        sys.exit(f'Required expected columns missing; no calculation performed: {missing}')

    # Official scope: T20I plus IPL competitions explicitly named IPL YYYY.
    comps['fmt']=comps.match_format.str.strip().str.upper()
    comps['comp_name_clean']=comps.comp_name.str.strip()
    comps['season_clean']=comps.season.str.strip()
    comps['scope_type']=np.where(comps.fmt.eq('T20I'),'T20I',np.where(
        comps.fmt.eq('T20') & comps.comp_name_clean.str.match(r'(?i)^IPL\s+\d{4}$'),'IPL','EXCLUDE'))
    matches['date_parsed']=pd.to_datetime(matches.match_date, errors='coerce')
    m=matches.merge(comps[['comp_id','comp_name_clean','fmt','season_clean','scope_type']],on='comp_id',how='left',validate='many_to_one')
    m['eligible']=m.scope_type.isin(['IPL','T20I']) & m.date_parsed.ge(pd.Timestamp('2022-01-01'))
    em=m.loc[m.eligible].copy()
    eligible_ids=set(em.match_id.astype(str))
    em.to_csv(out/'eligible_matches.csv',index=False)

    bb['match_id']=bb.match_id.astype(str)
    bb['innings_n']=pd.to_numeric(bb.innings,errors='coerce')
    num(bb,['overid','over_ball_num','innings_ball_num','runs','extras1_runs','extras_2_runs'])
    bb['extra_label']=bb.extras_type.astype(str).str.strip().str.lower()
    bb['wicket']=boolcol(bb.is_wicket).astype(int)
    for c in ['is_four','is_six']:
        if c in bb: bb[c+'_b']=boolcol(bb[c]).astype(int)
        else: bb[c+'_b']=0
    b=bb[bb.match_id.isin(eligible_ids)].copy()
    b=b.merge(em[['match_id','comp_id','match_date','comp_name_clean','scope_type','season_clean','revised_target','first_inn_revised_overs','second_inn_revised_overs']],on='match_id',how='left',validate='many_to_one')
    b=b.sort_values(['match_id','innings_n','innings_ball_num','overid','over_ball_num'],kind='mergesort').reset_index(drop=True)
    # Provider field candidates: report all run combinations before selecting a meaning.
    b['runs0']=b.runs.fillna(0)
    b['ex1']=b.extras1_runs.fillna(0)
    b['ex2']=b.extras_2_runs.fillna(0)
    b['team_runs_candidate']=b.runs0+b.ex1+b.ex2
    b['illegal_candidate']=b.extra_label.isin(['wide','overstep','no-ball','noball','no ball'])
    b['legal_candidate']=~b.illegal_candidate

    # Join / coverage checks.
    all_match_ids=set(matches.match_id.astype(str)); all_ball_ids=set(bb.match_id.astype(str))
    eligible_bb_ids=set(b.match_id.astype(str))
    duplicate_positions=b.duplicated(['match_id','innings_n','innings_ball_num'],keep=False)
    # Per-innings diagnostics.
    rows=[]
    for (mid,inn),g in b.groupby(['match_id','innings_n'],sort=False):
        rows.append({
          'match_id':mid,'innings':int(inn) if pd.notna(inn) else None,
          'competition_type':g.scope_type.iloc[0],'competition':g.comp_name_clean.iloc[0],
          'date':g.match_date.iloc[0],'delivery_rows':len(g),
          'distinct_innings_ball_num':int(g.innings_ball_num.nunique()),
          'innings_ball_num_min':g.innings_ball_num.min(),'innings_ball_num_max':g.innings_ball_num.max(),
          'overid_min':g.overid.min(),'overid_max':g.overid.max(),'over_ball_num_max':g.over_ball_num.max(),
          'candidate_legal_balls':int(g.legal_candidate.sum()),'candidate_illegal_balls':int(g.illegal_candidate.sum()),
          'sum_runs_only':float(g.runs0.sum()),'sum_runs_plus_ex1':float((g.runs0+g.ex1).sum()),
          'sum_runs_plus_ex1_ex2':float(g.team_runs_candidate.sum()),'wicket_flags':int(g.wicket.sum()),
          'max_ballnum_equals_rows':bool(pd.notna(g.innings_ball_num.max()) and int(g.innings_ball_num.max())==len(g)),
          'max_ballnum_equals_legal_count':bool(pd.notna(g.innings_ball_num.max()) and int(g.innings_ball_num.max())==int(g.legal_candidate.sum())),
          'extras_labels':'|'.join(sorted(x for x in g.extra_label.unique() if x and x!='none'))})
    idiag=pd.DataFrame(rows)
    idiag.to_csv(out/'innings_diagnostics.csv',index=False)

    # Extras/run accounting by type plus sample rows for manual semantics review.
    exsum=b.groupby('extra_label',dropna=False).agg(
        deliveries=('match_id','size'),runs_sum=('runs0','sum'),extras1_sum=('ex1','sum'),extras2_sum=('ex2','sum'),
        total_candidate_sum=('team_runs_candidate','sum'),candidate_legal=('legal_candidate','sum'),wickets=('wicket','sum')).reset_index()
    exsum.to_csv(out/'extras_accounting_by_type.csv',index=False)
    sample_cols=[c for c in ['match_id','innings','overid','over_ball_num','innings_ball_num','striker_name','bowler_name','runs','extras_type','extras1_runs','extras_2_runs','is_four','is_six','is_wicket','dismissal_type','dismissed_player_id'] if c in b]
    b.loc[b.extra_label.ne('') & b.extra_label.ne('none'),sample_cols].head(100).to_csv(out/'delivery_semantics_samples.csv',index=False)

    # Candidate wicket attribution. Verify before final use.
    dis=b.get('dismissal_type',pd.Series('',index=b.index)).fillna('').astype(str).str.strip().str.lower()
    credit={'bowled','caught','caught and bowled','lbw','stump','stumped','hit wicket'}
    b['candidate_bowler_wicket']=b.wicket.eq(1)&dis.isin(credit)

    # Make state as-of BEFORE each row; use the current row as the event after state.
    b['pre_score']=0.0; b['pre_wickets']=0; b['legal_before']=0
    states=[]
    for (mid,inn),idx in b.groupby(['match_id','innings_n'],sort=False).groups.items():
        g=b.loc[list(idx)].sort_values(['innings_ball_num','overid','over_ball_num'],kind='mergesort').copy()
        g['pre_score']=g.team_runs_candidate.cumsum().shift(1,fill_value=0)
        g['pre_wickets']=g.wicket.cumsum().shift(1,fill_value=0)
        g['legal_before']=g.legal_candidate.astype(int).cumsum().shift(1,fill_value=0)
        states.append(g)
    s=pd.concat(states,ignore_index=True) if states else b.iloc[:0].copy()
    # First innings par profile, pooled by eligible competition type and legal ball index.
    first=s[s.innings_n.eq(1)].copy()
    if len(first):
        par=first.groupby(['scope_type','legal_before']).pre_score.mean().rename('par_score')
        first=first.join(par,on=['scope_type','legal_before'])
        first['par_deficit']=((first.par_score-first.pre_score)/first.par_score.replace(0,np.nan)).clip(0,1).fillna(0)
        first['stage']= (first.legal_before/120).clip(0,1)
        first['wicket_scarcity']=(first.pre_wickets/10).clip(0,1)
        first['bat_pressure_probe']=.45*first.wicket_scarcity+.35*first.par_deficit+.20*first.stage
        current_rr=first.pre_score*6/first.legal_before.replace(0,np.nan)
        rr_pct=current_rr.groupby(first.scope_type).rank(pct=True).fillna(0)
        wickets_left=(10-first.pre_wickets).clip(0,10)/10
        first['bowl_pressure_probe']=.50*rr_pct+.30*wickets_left+.20*first.stage
        first['rrr_probe']=np.nan
        s=s.merge(first[['match_id','innings_n','innings_ball_num','bat_pressure_probe','bowl_pressure_probe','par_score','par_deficit']],on=['match_id','innings_n','innings_ball_num'],how='left')
    else:
        s['bat_pressure_probe']=np.nan;s['bowl_pressure_probe']=np.nan

    # Chase state: ordinary full-length matches only. Revised fields are not interpreted.
    first_tot=idiag[idiag.innings.eq(1)][['match_id','sum_runs_plus_ex1_ex2']].rename(columns={'sum_runs_plus_ex1_ex2':'first_innings_total_candidate'})
    s=s.merge(first_tot,on='match_id',how='left')
    if 'second_inn_revised_overs' in s: revised_overs=s.second_inn_revised_overs.astype(str).str.strip().ne('')
    else: revised_overs=pd.Series(False,index=s.index)
    if 'revised_target' in s: revised_target=s.revised_target.astype(str).str.strip().ne('')
    else: revised_target=pd.Series(False,index=s.index)
    chase=s.innings_n.eq(2)&~revised_overs&~revised_target&s.first_innings_total_candidate.notna()
    ch=s.loc[chase].copy()
    if len(ch):
        ch['target_candidate']=ch.first_innings_total_candidate+1
        ch['balls_left_candidate']=(120-ch.legal_before).clip(lower=1)
        ch['runs_needed_candidate']=(ch.target_candidate-ch.pre_score).clip(lower=0)
        ch['rrr_probe']=ch.runs_needed_candidate*6/ch.balls_left_candidate
        ch['wickets_left']=(10-ch.pre_wickets).clip(lower=0)
        ch['urgency_pct']=ch.groupby('scope_type').rrr_probe.rank(pct=True)
        ch['stage']=(ch.legal_before/120).clip(0,1)
        ch['bat_pressure_probe']=.50*ch.urgency_pct+.30*(1-ch.wickets_left/10)+.20*ch.stage
        current_rr=ch.pre_score*6/ch.legal_before.replace(0,np.nan)
        ratio=(ch.rrr_probe+.5)/(current_rr+.5)
        closeness=np.exp(-np.abs(np.log(ratio.clip(lower=1e-8))))
        ch['bowl_pressure_probe']=.50*closeness.fillna(0)+.30*ch.stage+.20*(ch.wickets_left/10)
        ch['par_score']=np.nan;ch['par_deficit']=np.nan
        s=s.merge(ch[['match_id','innings_n','innings_ball_num','bat_pressure_probe','bowl_pressure_probe','rrr_probe','target_candidate','balls_left_candidate']],on=['match_id','innings_n','innings_ball_num'],how='left',suffixes=('','_chase'))
        # Merge may create suffixes if columns already exist; consolidate candidate values.
        for col in ['bat_pressure_probe','bowl_pressure_probe','rrr_probe']:
            alt=col+'_chase'
            if alt in s:
                s[col]=s[alt].combine_first(s[col]);s.drop(columns=[alt],inplace=True)
    else:
        s['rrr_probe']=np.nan;s['target_candidate']=np.nan;s['balls_left_candidate']=np.nan

    # Candidate high-pressure opportunity counts: top quartile within format x innings.
    coverage=[]; player_rows=[]
    for role, col, player_id, player_name in [
        ('batter','bat_pressure_probe','striker_id','striker_name'),
        ('bowler','bowl_pressure_probe','bowler_id','bowler_name')]:
        for (fmt,inn),g in s.groupby(['scope_type','innings_n'],dropna=False):
            if inn not in [1,2] or col not in g: continue
            vals=g[col].dropna()
            if len(vals)<10: continue
            q75=float(vals.quantile(.75));q50=float(vals.quantile(.50))
            hi=g[g[col]>=q75];normal=g[g[col]<=q50]
            hv=hi.groupby(player_id).agg(high_balls=('match_id','size'),high_matches=('match_id','nunique'))
            coverage.append({
                'role':role,'competition':fmt,'innings':int(inn),'deliveries_scored':int(len(vals)),
                'provisional_high_pressure_balls':int(len(hi)),'provisional_normal_balls':int(len(normal)),
                'players_with_high_pressure_sample':int(hi[player_id].nunique()),
                'players_with_10plus_high_balls':int((hv.high_balls>=10).sum()),
                'players_with_20plus_high_balls':int((hv.high_balls>=20).sum()),
                'players_with_30plus_high_balls':int((hv.high_balls>=30).sum()),
                'players_with_10plus_high_matches':int((hv.high_matches>=10).sum())})
            for pid,pg in g.groupby(player_id):
                h=pg[pg[col]>=q75];n=pg[pg[col]<=q50]
                player_rows.append({'role':role,'player_id':pid,'player_name':pg[player_name].iloc[0],
                    'competition':fmt,'innings':int(inn),'high_pressure_balls_probe':len(h),
                    'high_pressure_matches_probe':h.match_id.nunique(),
                    'normal_balls_probe':len(n),'normal_matches_probe':n.match_id.nunique()})
    pd.DataFrame(coverage).to_csv(out/'provisional_pressure_coverage.csv',index=False)
    pp=pd.DataFrame(player_rows)
    pp.to_csv(out/'player_pressure_probe_counts.csv',index=False)
    # Save provisional delivery-level context for inspection; it is not final model output.
    cols=[c for c in ['match_id','match_date','scope_type','innings','innings_n','innings_ball_num','overid',
      'striker_id','striker_name','bowler_id','bowler_name','pre_score','pre_wickets','legal_before',
      'rrr_probe','target_candidate','balls_left_candidate','par_score','par_deficit',
      'bat_pressure_probe','bowl_pressure_probe','runs','extras_type','extras1_runs','extras_2_runs','is_wicket'] if c in s]
    s[cols].to_csv(out/'provisional_delivery_pressure_probe.csv',index=False)

    # Details dictionary.
    detail={
      'generated_at':datetime.now().astimezone().isoformat(),'source_folder':str(root),'source_csvs_modified':False,
      'rows':{'ball_by_ball':len(bb),'matches':len(matches),'competitions':len(comps),'mapping':len(mapping),'playing_eleven':len(xi),'teams':len(teams)},
      'competition_scope':{
        'competitions_by_format':comps.fmt.value_counts().to_dict(),
        'eligible_competitions':int(comps.scope_type.isin(['IPL','T20I']).sum()),
        'eligible_ipl_competitions':int(comps.scope_type.eq('IPL').sum()),'eligible_t20i_competitions':int(comps.scope_type.eq('T20I').sum()),
        'eligible_matches':len(em),'eligible_ipl_matches':int(em.scope_type.eq('IPL').sum()),'eligible_t20i_matches':int(em.scope_type.eq('T20I').sum()),
        'eligible_ball_rows':len(b),'eligible_matches_without_ball_data':sorted(eligible_ids-all_ball_ids),
        'eligible_matches_not_in_playing_xi':sorted(eligible_ids-set(xi.match_id.astype(str)))},
      'delivery_sequence':{
        'duplicate_match_innings_innings_ball_positions':int(duplicate_positions.sum()),
        'innings_distribution':counts(b.innings_n,10),'overid_distribution':counts(b.overid,30),
        'over_ball_num_distribution':counts(b.over_ball_num,30),
        'innings_ball_num_min':float(b.innings_ball_num.min()),'innings_ball_num_max':float(b.innings_ball_num.max()),
        'innings_groups':len(idiag),'innings3_4_delivery_rows':int(b.innings_n.isin([3,4]).sum()),
        'innings_groups_max_ballnum_equals_delivery_rows':int(idiag.max_ballnum_equals_rows.sum()),
        'innings_groups_max_ballnum_equals_candidate_legal_balls':int(idiag.max_ballnum_equals_legal_count.sum())},
      'runs_extras':{'by_extras_type':exsum.to_dict('records'),
        'interpretation':'Inspect delivery_semantics_samples.csv. Candidate team runs = runs + extras1_runs + extras_2_runs; not approved as official until verified.'},
      'wickets':{'wicket_flags':int(b.wicket.sum()),'candidate_bowler_credited_wickets':int(b.candidate_bowler_wicket.sum()),
        'dismissal_type_counts':counts(b.get('dismissal_type',pd.Series('',index=b.index)),30),
        'candidate_credit_rule':'bowled, caught, caught and bowled, lbw, stump/stumped, hit wicket; verify provider semantics.'},
      'revised_fields':{
        'first_inn_revised_overs_nonblank':int(em.first_inn_revised_overs.astype(str).str.strip().ne('').sum()),
        'second_inn_revised_overs_nonblank':int(em.second_inn_revised_overs.astype(str).str.strip().ne('').sum()),
        'revised_target_nonblank':int(em.revised_target.astype(str).str.strip().ne('').sum()),
        'first_inn_revised_overs_values':counts(em.first_inn_revised_overs,30),
        'second_inn_revised_overs_values':counts(em.second_inn_revised_overs,30),
        'revised_target_examples':em.loc[em.revised_target.astype(str).str.strip().ne(''),['match_id','comp_name_clean','match_date','first_inn_revised_overs','second_inn_revised_overs','revised_target']].head(30).to_dict('records'),
        'warning':'Units of revised_overs fields are not assumed.'},
      'pressure_probe_coverage':coverage,
      'join_checks':{
        'eligible_striker_ids_missing_competition_mapping':int(len(set(b.striker_id.astype(str))-set(mapping.player_id.astype(str)))),
        'eligible_bowler_ids_missing_competition_mapping':int(len(set(b.bowler_id.astype(str))-set(mapping.player_id.astype(str)))),
        'eligible_delivery_team_ids_missing_teams_table':int(len((set(b.batting_team_id.astype(str))|set(b.bowling_team_id.astype(str)))-set(teams.team_id.astype(str))))},
      'assumptions_and_warnings':[
        'Pressure scores are provisional diagnostic proxies, not final ratings or rankings.',
        'Candidate legal-ball rule treats wide and Overstep as illegal; validate against actual examples/provider convention.',
        'Candidate team runs use runs + extras1_runs + extras_2_runs; inspect samples before confirming.',
        'Chase probe uses 120 legal balls and target = first-innings candidate total + 1; revised matches excluded.',
        'First-innings par curve is a descriptive historical average by competition type and legal-ball stage; it is not a causal or predictive model.',
        'Innings 3/4 are reported separately and are not included in the first/second-innings pressure probe.',
        'Role-specific pressure proxies differ: batting pressure and bowling challenge should not be assumed identical.'
      ]}

    # Produce readable report.
    L=['# SupeRR Selector S2 — Pressure Feasibility Diagnostic','',f"Generated: {detail['generated_at']}",
       'Source CSVs were read only. This is a diagnostic, not final rankings.','',
       '## 1. Eligible competition scope','',
       f"- Eligible competitions: **{detail['competition_scope']['eligible_competitions']}** (IPL {detail['competition_scope']['eligible_ipl_competitions']}; T20I {detail['competition_scope']['eligible_t20i_competitions']})",
       f"- Eligible matches: **{detail['competition_scope']['eligible_matches']}** (IPL {detail['competition_scope']['eligible_ipl_matches']}; T20I {detail['competition_scope']['eligible_t20i_matches']})",
       f"- Eligible delivery rows: **{detail['competition_scope']['eligible_ball_rows']:,}**",
       f"- Eligible matches without ball-by-ball records: **{len(detail['competition_scope']['eligible_matches_without_ball_data'])}**",
       '', 'Scope rule: T20I competitions, plus T20 competitions named exactly like `IPL YYYY`, from match dates on/after 2022-01-01.','',
       '## 2. Delivery ordering and innings structure','',
       f"- Duplicate (match, innings, innings_ball_num) positions: **{detail['delivery_sequence']['duplicate_match_innings_innings_ball_positions']}**",
       f"- Innings groups: **{detail['delivery_sequence']['innings_groups']}**",
       f"- Innings 3/4 delivery rows: **{detail['delivery_sequence']['innings3_4_delivery_rows']}**",
       f"- Innings where max innings_ball_num equals delivery rows: **{detail['delivery_sequence']['innings_groups_max_ballnum_equals_delivery_rows']} / {detail['delivery_sequence']['innings_groups']}**",
       f"- Innings where max innings_ball_num equals candidate legal balls: **{detail['delivery_sequence']['innings_groups_max_ballnum_equals_candidate_legal_balls']} / {detail['delivery_sequence']['innings_groups']}**",
       '', 'See `innings_diagnostics.csv` for all innings. The diagnostics will help determine whether `innings_ball_num` counts every record or legal balls.','',
       '## 3. Runs and extras interpretation','',
       'Three candidate totals are compared: `runs`; `runs + extras1_runs`; and `runs + extras1_runs + extras_2_runs`. Inspect the sample CSV before deciding which represents team runs.','',
       '| Extras type | Deliveries | Runs | Extras1 | Extras2 | Runs + both extras | Candidate legal |','|---|---:|---:|---:|---:|---:|---:|']
    for r in exsum.to_dict('records'):
        L.append(f"| {r['extra_label'] or '(blank)'} | {r['deliveries']} | {r['runs_sum']} | {r['extras1_sum']} | {r['extras2_sum']} | {r['total_candidate_sum']} | {r['candidate_legal']} |")
    L += ['', 'Candidate illegal labels are wide, overstep, no-ball/noball/no ball. Confirm this using `delivery_semantics_samples.csv`.','',
       '## 4. Wicket attribution','',
       f"- Wicket flags: **{int(b.wicket.sum())}**",
       f"- Candidate bowler-credited wickets: **{int(b.candidate_bowler_wicket.sum())}**",
       'Candidate credited types: bowled, caught, caught and bowled, LBW, stumped, hit wicket. Validate with the provider convention.','',
       '## 5. Revised-match fields','',
       f"- First-innings revised-overs field nonblank: **{detail['revised_fields']['first_inn_revised_overs_nonblank']}**",
       f"- Second-innings revised-overs field nonblank: **{detail['revised_fields']['second_inn_revised_overs_nonblank']}**",
       f"- Revised target nonblank: **{detail['revised_fields']['revised_target_nonblank']}**",
       'The script does not assume whether revised-overs values are expressed as overs, balls, or another encoding. Inspect the detail JSON.','',
       '## 6. Provisional pressure-opportunity counts','',
       'Batting and bowling get separate provisional pressure proxies. First innings uses wicket/resource context, innings stage, and a historical par-score curve for batting; the bowling proxy uses opposition scoring threat, resources and stage. Chase batting uses required-rate urgency, wickets and balls; chase bowling uses closeness of current scoring rate to required rate plus stage/resources. High pressure = top quartile within competition type × innings; normal = bottom half.','',
       '| Role | Competition | Innings | Scored balls | High-pressure balls | Normal balls | Players 10+ high balls | 20+ | 30+ | Players 10+ high-pressure matches |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in coverage:
        L.append(f"| {r['role']} | {r['competition']} | {r['innings']} | {r['deliveries_scored']} | {r['provisional_high_pressure_balls']} | {r['provisional_normal_balls']} | {r['players_with_10plus_high_balls']} | {r['players_with_20plus_high_balls']} | {r['players_with_30plus_high_balls']} | {r['players_with_10plus_high_matches']} |")
    L += ['', 'These are feasibility counts, not approved ranking samples. See `player_pressure_probe_counts.csv` for per-player opportunity counts and `provisional_delivery_pressure_probe.csv` for delivery-level states.','',
       '## 7. Join checks','',
       f"- Eligible striker IDs missing from player mapping: **{detail['join_checks']['eligible_striker_ids_missing_competition_mapping']}**",
       f"- Eligible bowler IDs missing from player mapping: **{detail['join_checks']['eligible_bowler_ids_missing_competition_mapping']}**",
       f"- Eligible delivery team IDs missing teams table: **{detail['join_checks']['eligible_delivery_team_ids_missing_teams_table']}**",'',
       '## 8. Assumptions and warnings','']
    L += ['- '+x for x in detail['assumptions_and_warnings']]
    L += ['', '## 9. Outputs','', '- `pressure_feasibility_report.md`','- `pressure_feasibility_details.json`','- `eligible_matches.csv`','- `innings_diagnostics.csv`','- `extras_accounting_by_type.csv`','- `delivery_semantics_samples.csv`','- `player_pressure_probe_counts.csv`','- `provisional_pressure_coverage.csv`','- `provisional_delivery_pressure_probe.csv`','']
    (out/'pressure_feasibility_report.md').write_text('\n'.join(L),encoding='utf-8')
    (out/'pressure_feasibility_details.json').write_text(json.dumps(detail,indent=2,ensure_ascii=False,default=lambda x: x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
    print('\nDIAGNOSTIC COMPLETE')
    for f in ['pressure_feasibility_report.md','pressure_feasibility_details.json','eligible_matches.csv','innings_diagnostics.csv','extras_accounting_by_type.csv','delivery_semantics_samples.csv','player_pressure_probe_counts.csv','provisional_pressure_coverage.csv','provisional_delivery_pressure_probe.csv']:
        print(out/f)
    print('\nShare pressure_feasibility_report.md first. Keep the CSV diagnostics available.')

if __name__=='__main__':
    main()
