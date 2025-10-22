# sim_cards.py
import random
import itertools
import math
import copy
import pandas as pd
from functools import lru_cache
from tqdm import tqdm

random.seed(87878787)

# ---------------------------
# CARD DB
# ---------------------------

CARD_DB = {
    # STARTERS
    'oath':       {'class':'S', 'base':4.0, 'cost':0, 'max_per_deck':3, 'effects': ['lock']},
    'ignite':     {'class':'S', 'base':2.0, 'cost':0, 'max_per_deck':3, 'effects': ['ignite_modes']},
    'brand1':     {'class':'S', 'base':3.0, 'cost':0, 'max_per_deck':3, 'effects': ['make_token_next']},
    'brand2':     {'class':'S', 'base':2.0, 'cost':0, 'max_per_deck':3, 'effects': ['make_token_next']},
    'regular':    {'class':'S', 'base':3.0, 'cost':0, 'max_per_deck':6, 'effects': []},
    'blue':       {'class':'S', 'base':1.0, 'cost':0, 'max_per_deck':21,'effects':['double_starter_create_resource_penalize_next']},
    'dnd':        {'class':'S', 'base':4.0, 'cost':1, 'max_per_deck':3, 'effects': ['dnd_duplication']},
    'spreading':  {'class':'S', 'base':3.0, 'cost':1, 'max_per_deck':3, 'effects': ['spreading_buff']},

    # EXTENDERS
    'wrath':      {'class':'E', 'base':7.0, 'cost':0, 'max_per_deck':1, 'effects': []},
    'headlong':   {'class':'E', 'base':4.0, 'cost':0, 'max_per_deck':3, 'effects': []},
    'devotion':   {'class':'E', 'base':4.0, 'cost':1, 'max_per_deck':6, 'effects': ['makes_token']},
    'loyalty':    {'class':'E', 'base':3.0, 'cost':0, 'max_per_deck':3, 'effects': ['makes_token']},
    'filler':     {'class':'E', 'base':3.0, 'cost':0, 'max_per_deck':9, 'effects': []},

    # FINISHERS
    'hunt':       {'class':'F', 'base':5.0, 'cost':1, 'max_per_deck':3, 'effects': ['makes_token']},
    'breaking':   {'class':'F', 'base':6.5, 'cost':1, 'max_per_deck':3, 'effects': []},
    'burst':      {'class':'F', 'base':5.0, 'cost':0, 'max_per_deck':3, 'effects': []},
    'march':      {'class':'F', 'base':4.0, 'cost':0, 'max_per_deck':3, 'effects': ['march_token_chain']},

    # GENERICS
    'rabble':     {'class':'G', 'base':4.0, 'cost':0, 'max_per_deck':15, 'effects': []},
    'blood':      {'class':'G', 'base':4.0, 'cost':0, 'max_per_deck':3, 'effects': ['blood_token_effect']},
    'art':        {'class':'G', 'base':5.0, 'cost':1, 'max_per_deck':6, 'effects': ['art_ends_turn_token_boost']},
    'cnc':        {'class':'G', 'base':7.5, 'cost':2, 'max_per_deck':3, 'effects': ['ends_turn']},
    'snatch':     {'class':'G', 'base':4.0, 'cost':0, 'max_per_deck':3, 'effects': ['ends_turn','snatch_variance']},
    'estrike':    {'class':'G', 'base':7.5, 'cost':0, 'max_per_deck':3, 'effects': ['estrike_discard']},
    'rake':       {'class':'G', 'base':0.0, 'cost':0, 'max_per_deck':3, 'effects': ['rake_buff']},

    # XTRAS
    'throw_dagger': {'class':'X', 'base':1.0, 'cost':0, 'max_per_deck':3, 'effects': ['X_modes']},

}

# ---------------------------
# DECK BUILDER
# ---------------------------

def build_deck_from_counts(counts):
    deck = []
    total = 0
    for name, cnt in counts.items():
        if name not in CARD_DB:
            raise ValueError(f"Unknown card: {name}")
        max_allowed = CARD_DB[name].get('max_per_deck', 3)
        if cnt > max_allowed:
            raise ValueError(f"Card {name} exceeds allowed copies ({cnt} > {max_allowed})")
        for i in range(cnt):
            card = dict(CARD_DB[name])
            card['name'] = name
            card['uid'] = f"{name}_{i+1}"
            deck.append(card)
            total += 1
    if total != 60:
        raise ValueError(f"Deck size must be 60 cards: current size {total}")
    random.shuffle(deck)
    return deck

# ---------------------------
# DRAW / HAND
# ---------------------------
#placeholder hand probabilities before I make anything to do with blocking on oppo turn
hand_probs = {0:0.005, 1:0.005, 2:0.02, 3:0.02, 4:0.85, 5:0.10}
hand_sizes = list(hand_probs.keys())
hand_weights = list(hand_probs.values())

def draw_hand(deck):
    n = random.choices(hand_sizes, weights=hand_weights, k=1)[0]
    n = min(n, len(deck))
    return random.sample(deck, k=n) if n>0 else []

# ---------------------------
# DETERMINISTIC SOLVER
# ---------------------------
def best_sequence_for_hand(hand, game_state, depth_limit=8):

    uid2card = {c['uid']: c for c in hand}
    start_uids = tuple([c['uid'] for c in hand])

    def snatch_value():
        return 0.2*7 + 0.8*4

    def score_outcome(points, sef, tokens, resources, resources_generated_delta):
        bonus = 3.0 if sef >= 3 else 0.0
        return points + bonus + 0.01*tokens + 0.005*resources - 0.001*resources_generated_delta

    initial_resources = game_state.get('resources', 0)
    initial_resources_generated_total = game_state.get('resources_generated_total', 0)
    initial_tokens = game_state.get('tokens', 0)

    @lru_cache(maxsize=None)
    def recurse(remaining_uids, tokens, resources, resources_generated_total, blood_counter,
                starter_played, sef_played, cards_played, penalize_this_turn, penalty_taken_this_turn,
                ignite_reduce_next_cost, rake_active, spread_active, token_created_this_turn):
        best = {'points':0.0,'sef':0,'tokens':tokens,'resources':resources,
                'resources_generated_delta':0,'penalize_next_turn_flag':penalize_this_turn,'cards_played':cards_played}
        best_score = score_outcome(0,0,tokens,resources,0)
        if cards_played >= depth_limit:
            return best

        remaining = [uid2card[uid] for uid in remaining_uids]
        def remove_uid(uids, uid):
            lst = list(uids)
            lst.remove(uid)
            return tuple(lst)

        # passive resource gen
        if (not penalty_taken_this_turn) and (resources_generated_total < 3):
            res_branch = recurse(remaining_uids, tokens, resources+1, resources_generated_total+1,
                                 blood_counter, starter_played, sef_played, cards_played,
                                 penalize_this_turn, penalty_taken_this_turn,
                                 ignite_reduce_next_cost, rake_active, spread_active, token_created_this_turn)
            sc = score_outcome(res_branch['points'], res_branch['sef'], res_branch['tokens'],
                               res_branch['resources'], res_branch['resources_generated_delta']+1)
            if sc > best_score:
                best_score = sc
                best = res_branch.copy()
                best['resources_generated_delta'] += 1

        # discard for resource (discard any card)
        for c in remaining:
            uid = c['uid']
            new_remaining = remove_uid(remaining_uids, uid)
            branch = recurse(new_remaining, tokens, resources+1, resources_generated_total,
                             blood_counter, starter_played, sef_played, cards_played+1,
                             penalize_this_turn, True, ignite_reduce_next_cost,
                             rake_active, spread_active, token_created_this_turn)
            sc = score_outcome(branch['points'], branch['sef'], branch['tokens'], branch['resources'],
                               branch['resources_generated_delta'])
            if sc > best_score:
                best_score = sc
                best = branch.copy()

        # iterate possible plays
        for c in remaining:
            uid = c['uid']
            name = c['name']
            base = c['base']
            cost = c.get('cost',0)
            effects = c.get('effects',[])
            effective_cost = max(0, cost - (1 if blood_counter>0 else 0) - ignite_reduce_next_cost)
            if effective_cost > resources:
                continue

            modes = []
            def M(**kw): d={'mode_name':'normal','delta_points':0.0,'delta_sef':0,'delta_tokens':0,'delta_resources':0,
                            'penalize':False,'ends_turn':False,'draw':0,'use_token':False,'grant_next_token':False}; d.update(kw); return d

            # --- card-specific modes ---
            if name=='oath':
                modes.append(M(delta_points=base,delta_sef=1,lockset=True))
            elif name=='ignite':
                modes.append(M(mode_name='ignite_boost_next',delta_points=2.0,delta_sef=1,ignite_reduce_next_cost_set=1))
                modes.append(M(mode_name='ignite_double_starter',delta_points=3.0,delta_sef=2))
            elif name in ['brand1','brand2']:
                modes.append(M(delta_points=base,delta_sef=1,grant_next_token=True))
            elif name=='regular':
                modes.append(M(delta_points=base,delta_sef=1))
            elif name=='blue':
                modes.append(M(delta_points=2.0,delta_sef=2,delta_resources=1,penalize=True))
            elif name=='wrath':
                modes.append(M(delta_points=base,delta_sef=1))
            elif name=='headlong':
                modes.append(M(delta_points=base,delta_sef=1))
            elif name=='devotion':
                modes.append(M(delta_points=base,delta_sef=1,delta_tokens=1,delta_resources=-1))
            elif name=='loyalty':
                modes.append(M(delta_points=base,delta_sef=1,delta_tokens=1))
            elif name=='filler':
                modes.append(M(delta_points=base,delta_sef=1))
            elif name=='hunt':
                modes.append(M(delta_points=base,delta_sef=1,delta_tokens=1,delta_resources=-1))
            elif name=='breaking':
                modes.append(M(delta_points=base,delta_sef=1,delta_resources=-1))
            elif name=='snatch':
                modes.append(M(delta_points=snatch_value(),delta_sef=1,ends_turn=True))
            elif name=='burst':
                modes.append(M(delta_points=base,delta_sef=1))
            elif name=='rabble':
                modes.append(M(delta_points=base,delta_sef=1))
            elif name=='blood':
                modes.append(M(mode_name='blood_no_token',delta_points=base,ends_turn=True))
                if tokens>0:
                    modes.append(M(mode_name='blood_with_token',delta_points=base,delta_sef=1,
                                   delta_tokens=-1,blood_set_counter=3))
            elif name=='art':
                modes.append(M(mode_name='art_no_token',delta_points=base,delta_resources=-1,ends_turn=True))
                if tokens>0:
                    modes.append(M(mode_name='art_with_token',delta_points=7.0,delta_resources=-1,
                                   delta_tokens=-1,ends_turn=True))
            elif name=='cnc':
                modes.append(M(delta_points=base,delta_resources=-2,ends_turn=True))
            elif name=='throw_dagger':
                modes.append(M(mode_name='throw_as_is',delta_points=1.0,delta_sef=0,draw=1))
                modes.append(M(mode_name='throw_double',delta_points=2.0,delta_sef=2,penalize=True))
            elif name=='dnd':

                modes.append(M(mode_name='dnd_play',delta_points=base + 0.1*4.0, delta_sef=1 + 0.1, delta_resources=-1))
            elif name=='spreading':
                modes.append(M(mode_name='spreading',delta_points=base, delta_sef=1, delta_resources=-1, set_spread=True))
            elif name=='march':
                modes.append(M(mode_name='march', delta_points=base, delta_sef=1, ends_turn=True, conditional_no_end_if_token_created=True))
            elif name=='estrike':
                pass
            elif name=='rake':
                modes.append(M(mode_name='rake', delta_points=0.0, delta_sef=0, set_rake=True))
            else:
                continue

            # handle estrike separately
            if name == 'estrike':
                if len(remaining) < 2:
                    continue
                for discard_card in remaining:
                    if discard_card['uid'] == uid:
                        continue
                    new_remaining_after = remove_uid(remove_uid(remaining_uids, uid), discard_card['uid'])
                    mA = {'mode_name':'estrike_end','delta_points':7.5,'delta_sef':1,'delta_tokens':0,'delta_resources':0,'penalize':False,'ends_turn':True,'draw':0,'use_token':False}
                    mB = {'mode_name':'estrike_no_end','delta_points':5.0,'delta_sef':1,'delta_tokens':0,'delta_resources':0,'penalize':False,'ends_turn':False,'draw':0,'use_token':False}
                    for m in (mA, mB):
                        new_tokens = tokens + m.get('delta_tokens',0)
                        new_resources = resources + m.get('delta_resources',0)
                        new_blood = blood_counter
                        if blood_counter>0:
                            new_blood = max(0,blood_counter-1)
                        applied_points = m['delta_points']
                        if rake_active and (CARD_DB['estrike']['class'] in ['S','E','F','G']):
                            pass

                        new_ignite_cost = 0 if ignite_reduce_next_cost>0 else 0
                        branch = recurse(new_remaining_after, new_tokens, new_resources, resources_generated_total,
                                         new_blood, starter_played or (m['delta_sef']>0), sef_played + m['delta_sef'],
                                         cards_played+1, penalize_this_turn or m['penalize'],
                                         penalty_taken_this_turn, new_ignite_cost, rake_active, spread_active, token_created_this_turn)
                        total_points = applied_points + branch['points']
                        total_sef = m['delta_sef'] + branch['sef']
                        total_tokens = branch['tokens']
                        total_resources = branch['resources']
                        total_resources_delta = branch['resources_generated_delta']
                        penalize_flag = branch['penalize_next_turn_flag'] or m['penalize']
                        sc = score_outcome(total_points, total_sef, total_tokens, total_resources, total_resources_delta)
                        if sc > best_score:
                            best_score = sc
                            best = {'points':total_points,'sef':total_sef,'tokens':total_tokens,
                                    'resources':total_resources,'resources_generated_delta':total_resources_delta,
                                    'penalize_next_turn_flag':penalize_flag,'cards_played':branch['cards_played']+1}
                continue

            # normal modes handling
            for m in modes:
                delta_res = m['delta_resources']
                if delta_res < 0 and abs(delta_res) > resources:
                    continue

                new_remaining = remove_uid(remaining_uids, uid)
                new_tokens = tokens + m['delta_tokens']
                new_resources = resources + delta_res
                new_blood = blood_counter
                if 'blood_set_counter' in m:
                    new_blood = m['blood_set_counter']
                elif blood_counter>0:
                    new_blood = max(0,blood_counter-1)

                # update token_created_this_turn - important for calcing march
                created_token_flag = False
                if m.get('delta_tokens',0) > 0 and not m.get('grant_next_token', False):
                    created_token_flag = True

                new_token_created = token_created_this_turn or created_token_flag

                new_starter = starter_played or (m.get('delta_sef',0) > 0)
                new_cards_played = cards_played + 1
                new_penalize = penalize_this_turn or m.get('penalize', False)
                new_penalty_taken = penalty_taken_this_turn

                # compute ignite cost flag update
                new_ignite_cost = 0
                if m.get('ignite_reduce_next_cost_set', 0):
                    new_ignite_cost = 1
                elif ignite_reduce_next_cost > 0:
                    new_ignite_cost = 0

                # compute applied points including rake/spread adjustments
                applied_points = m.get('delta_points', 0.0)
                card_class = CARD_DB[name]['class'] if name in CARD_DB else CARD_DB[name].get('class', None)
                if rake_active and CARD_DB.get(name,{}).get('class') in ['S','E','F']:
                    applied_points += 1.0

                if spread_active and CARD_DB.get(name,{}).get('class') in ['S','E','F']:
                    if base <= sef_played:
                        applied_points += 1.0

                # Special handling for march
                ends_turn_flag = m.get('ends_turn', False)
                if m.get('conditional_no_end_if_token_created', False):
                    if token_created_this_turn:
                        ends_turn_flag = False

                new_rake_active = rake_active or m.get('set_rake', False)
                new_spread_active = spread_active or m.get('set_spread', False)

                branch = recurse(new_remaining, new_tokens, new_resources, resources_generated_total,
                                 new_blood, new_starter, sef_played + m.get('delta_sef',0), new_cards_played,
                                 new_penalize, new_penalty_taken, new_ignite_cost,
                                 new_rake_active, new_spread_active, new_token_created)
                total_points = applied_points + branch['points']
                total_sef = m.get('delta_sef',0) + branch['sef']
                total_tokens = branch['tokens']
                total_resources = branch['resources']
                total_resources_delta = branch['resources_generated_delta']
                penalize_flag = branch['penalize_next_turn_flag'] or new_penalize
                sc = score_outcome(total_points, total_sef, total_tokens, total_resources, total_resources_delta)
                if sc > best_score:
                    best_score = sc
                    best = {'points':total_points,'sef':total_sef,'tokens':total_tokens,
                            'resources':total_resources,'resources_generated_delta':total_resources_delta,
                            'penalize_next_turn_flag':penalize_flag,'cards_played':branch['cards_played']+1}

        # treat any card as starter (no token consumed) if no starter played yet (pitching, basically)
        if (not starter_played) and remaining:
            for c in remaining:
                uid = c['uid']
                new_remaining = remove_uid(remaining_uids, uid)
                branch = recurse(new_remaining, tokens, resources, resources_generated_total,
                                 blood_counter, True, sef_played+1, cards_played+1,
                                 penalize_this_turn, penalty_taken_this_turn,
                                 ignite_reduce_next_cost, rake_active, spread_active, token_created_this_turn)
                total_points = 1.0 + branch['points']
                total_sef = 1 + branch['sef']
                sc = score_outcome(total_points, total_sef, branch['tokens'], branch['resources'], branch['resources_generated_delta'])
                if sc > best_score:
                    best_score = sc
                    best = {'points':total_points,'sef':total_sef,'tokens':branch['tokens'],'resources':branch['resources'],
                            'resources_generated_delta':branch['resources_generated_delta'],
                            'penalize_next_turn_flag':branch['penalize_next_turn_flag'],'cards_played':branch['cards_played']+1}
        return best

    res = recurse(start_uids, initial_tokens, initial_resources, initial_resources_generated_total,
                  0, False, 0, 0, False, False, 0, False, False, False)
    res['resources_generated_delta'] = res.get('resources_generated_delta', 0)
    return res

# ---------------------------
# PLAY HAND WRAPPER
# ---------------------------
def play_hand(hand, game_state):
    best = best_sequence_for_hand(hand, game_state, depth_limit=max(6, len(hand)*2))
    points = best['points']
    if game_state.get('next_turn_penalty_from_prev', False):
        points = max(0.0, points - 1.0)
    return points, best['sef'], best['tokens'], best['resources'], game_state.get('resources_generated_total',0)+best['resources_generated_delta'], best['penalize_next_turn_flag'], best['cards_played']

# ---------------------------
# SIMULATION
# ---------------------------
def simulate_deck(counts, trials=200, turns=50, deck_name="deck"):
    deck = build_deck_from_counts(counts)
    tot_points=tot_sef=tot_pen_freq=tot_finishers=0
    for _ in tqdm(range(trials), desc=f"Sim {deck_name}", ncols=80):
        resources=0; resources_generated_total=0; tokens=0; next_turn_penalty_from_prev=False
        for t in range(turns):
            hand=draw_hand(deck)
            gs={'resources':resources,'resources_generated_total':resources_generated_total,
                'tokens':tokens,'next_turn_penalty_from_prev':next_turn_penalty_from_prev}
            pts,sef,tokens,resources,resources_generated_total,penalize_next_turn_flag,cards_played=play_hand(hand,gs)
            tot_points+=pts; tot_sef+=sef; tot_pen_freq+=(1 if penalize_next_turn_flag else 0)
            tot_finishers+=sum(1 for c in hand if c['name'] in ['hunt','breaking','snatch','burst','march'])
            next_turn_penalty_from_prev=penalize_next_turn_flag
    avg_ev=tot_points/(trials*turns)
    avg_sef=tot_sef/(trials*turns)
    avg_pen=tot_pen_freq/(trials*turns)
    avg_fin=tot_finishers/(trials*turns)
    return {'est_EV':round(avg_ev,4),'avg_SEF_per_turn':round(avg_sef,3),
            'next_turn_penalty_freq':round(avg_pen,3),'avg_finishers_per_turn':round(avg_fin,3)}

# ---------------------------
# Utilities: archetype + detailed simulation + analyzer
# ---------------------------

def get_archetype(counts):
    if counts.get('blue', 0) >= 10:
        return "Blue Heavy"
    if counts.get('devotion', 0) >= 4:
        return "magma+demonstrate"
    if counts.get('blood', 0) >= 3 and counts.get('art', 0) >= 3:
        return "Blood+art"
    if counts.get('dnd', 0) >= 1:
        return "DND"
    if counts.get('spreading', 0) >= 2:
        return "Spreading Combo"
    if counts.get('rake', 0) >= 3:
        return "Rake"
    # fallback
    return "Midrange"

def simulate_deck_detailed(counts, trials=200, turns=50):

    deck = build_deck_from_counts(counts)
    tot_points = tot_sef = tot_pen = tot_finishers = 0.0
    tot_resources_generated = 0.0
    tot_resources_spent_est = 0.0
    tot_resource_balance = 0.0
    tot_tokens_created = 0.0
    tot_tokens_spent = 0.0
    tot_cards_played = 0.0
    tot_free_turns = 0.0

    for _ in range(trials):
        resources = 0
        resources_generated_total = 0
        tokens = 0
        next_turn_penalty_from_prev = False
        prev_sef = 0
        for t in range(turns):
            prev_resources = resources
            prev_tokens = tokens
            prev_resources_generated_total = resources_generated_total

            hand = draw_hand(deck)
            pts, sef, tokens, resources, resources_generated_total, penalize_next_turn_flag, cards_played = play_hand(
                hand,
                {'resources': resources,
                 'resources_generated_total': resources_generated_total,
                 'tokens': tokens,
                 'next_turn_penalty_from_prev': next_turn_penalty_from_prev}
            )

            tot_points += pts
            tot_sef += sef
            tot_pen += (1 if penalize_next_turn_flag else 0)
            tot_finishers += sum(1 for c in hand if c['name'] in ['hunt', 'breaking', 'snatch', 'burst', 'march'])
            resources_generated_this_turn = resources_generated_total - prev_resources_generated_total
            tot_resources_generated += resources_generated_this_turn
            resources_spent_est = max(0, (prev_resources + resources_generated_this_turn) - resources) # this undercounts if cards granted resources
            tot_resources_spent_est += resources_spent_est
            tot_resource_balance += resources
            tokens_delta = tokens - prev_tokens
            if tokens_delta > 0:
                tot_tokens_created += tokens_delta
            elif tokens_delta < 0:
                tot_tokens_spent += (-tokens_delta)
            tot_cards_played += cards_played
            if prev_sef >= 3:
                tot_free_turns += 1
            next_turn_penalty_from_prev = penalize_next_turn_flag
            prev_sef = sef

    denom = trials * turns
    return {
        'est_EV': None,
        'avg_SEF_per_turn': tot_sef / denom,
        'avg_finishers_per_turn': tot_finishers / denom,
        'next_turn_penalty_freq': tot_pen / denom,
        'avg_resources_generated_per_turn': tot_resources_generated / denom,
        'avg_resources_spent_per_turn': tot_resources_spent_est / denom,
        'avg_resource_balance': tot_resource_balance / denom,
        'avg_tokens_created_per_turn': tot_tokens_created / denom,
        'avg_tokens_spent_per_turn': tot_tokens_spent / denom,
        'avg_token_usage_efficiency': (tot_tokens_created / (tot_tokens_spent + 1e-9)) if tot_tokens_spent > 0 else None,
        'avg_turn_length': tot_cards_played / denom,
        'free_turn_ratio': tot_free_turns / denom
    }

def analyze_deck(counts, deck_name=None, trials_main=200, trials_detailed=20, turns=50):

    base_stats = simulate_deck(counts, trials=trials_main, turns=turns, deck_name=(deck_name or "deck"))
    detail_stats = simulate_deck_detailed(counts, trials=trials_detailed, turns=turns)

    out = {}
    out.update(base_stats)
    out.update({
        'avg_resources_generated_per_turn': detail_stats.get('avg_resources_generated_per_turn'),
        'avg_resources_spent_per_turn': detail_stats.get('avg_resources_spent_per_turn'),
        'avg_resource_balance': detail_stats.get('avg_resource_balance'),
        'avg_tokens_created_per_turn': detail_stats.get('avg_tokens_created_per_turn'),
        'avg_tokens_spent_per_turn': detail_stats.get('avg_tokens_spent_per_turn'),
        'avg_token_usage_efficiency': detail_stats.get('avg_token_usage_efficiency'),
        'avg_turn_length': detail_stats.get('avg_turn_length'),
        'free_turn_ratio': detail_stats.get('free_turn_ratio'),
        # placeholders for stuff to add
        'ignite_synergy_success': None,
        'spreading_synergy_hits': None,
        'dnd_copy_frequency': None
    })
    return out

# ---------------------------
# PUT YOUR OWN SHIT IN HERE
# ---------------------------
if __name__ == "__main__":
    import pandas as pd
    import time

    # create decks with this template here, avoid estrike, its implementation sucks rn
    deck_counts_marchtempo = {
        'oath':1,'ignite':3,'brand1':3,'brand2':3,'regular':6,'blue':3,
        'dnd':0,'spreading':0,
        'wrath':1,'headlong':3,'devotion':3,'loyalty':3,'filler':6,
        'hunt':1,'breaking':3,'snatch':3,'burst':2,'march':3,
        'rabble':10,'blood':3,'art':0,'cnc':0,'rake':0,'throw_dagger':0, 'estrike':0
    }

    all_decks = {
            # put all decks here
            "marchtempo": deck_counts_marchtempo,
        }

    results = []

    for deck_name, deck_counts in all_decks.items():
        start = time.time()

        deck_size = sum(deck_counts.values())
        archetype = get_archetype(deck_counts)
        stats = analyze_deck(deck_counts, deck_name)

        extended_stats = {
            "deck_name": deck_name,
            "deck_size": deck_size,
            "deck_archetype": archetype,
            "est_EV": stats.get("est_EV"),
            "avg_SEF_per_turn": stats.get("avg_SEF_per_turn"),
            "avg_finishers_per_turn": stats.get("avg_finishers_per_turn"),
            "next_turn_penalty_freq": stats.get("next_turn_penalty_freq"),
            "avg_resources_generated_per_turn": stats.get("avg_resources_generated_per_turn"),
            "avg_resources_spent_per_turn": stats.get("avg_resources_spent_per_turn"),
            "avg_resource_balance": stats.get("avg_resource_balance"),
            "avg_tokens_created_per_turn": stats.get("avg_tokens_created_per_turn"),
            "avg_token_usage_efficiency": stats.get("avg_token_usage_efficiency"),
            "avg_turn_length": stats.get("avg_turn_length"),
            "free_turn_ratio": stats.get("free_turn_ratio"),
            "ignite_synergy_success": stats.get("ignite_synergy_success"), # not yet implemented
            "spreading_synergy_hits": stats.get("spreading_synergy_hits"), # not yet implemented
            "dnd_copy_frequency": stats.get("dnd_copy_frequency"), # not yet implemented
            "run_time_sec": round(time.time() - start, 2),
        }

        results.append(extended_stats)

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="est_EV", ascending=False)
    print("\n=== Simulation Summary ===\n")
    print(df_sorted.to_string(index=False))
    df_sorted.to_csv("deck_ev_summary.csv", index=False)
    print("\nSaved as deck_ev_summary.csv")
