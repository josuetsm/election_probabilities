import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import load_txt




def recode_party(df):
    return int(f"99{df['cod_zona'] - 6000}{df['can_orden']}") if df['cod_part'] == 999099 else df['cod_part']


def recode_pact(df):
    return int(f"99{df['cod_zona'] - 6000}{df['can_orden']}") if df['cod_pacto'] == 999 else df['cod_pacto']


def get_district_candidates(district, candidates, votes):
    """
    district: int district code 6001-6028
    candidates: pd.DataFrame candidates
    votes: pd.DataFrame with votes
    return: np.array of shape = (5, n) (n: number of district candidates)
    """
    candidates = candidates.copy()

    distric_votes = votes[votes['cod_zona'] == district].set_index('cod_ambito')
    distric_votes = distric_votes[['votos']]

    district_candidates = candidates.join(distric_votes, on='cod_cand', how='inner')
    district_candidates['cod_genero_int'] = district_candidates['cod_genero'].replace({'F': 0, 'M': 1})
    district_candidates['cod_part'] = district_candidates.apply(recode_party, axis=1)
    district_candidates['cod_pacto'] = district_candidates.apply(recode_pact, axis=1)

    district_candidates = district_candidates[['cod_cand', 'cod_genero_int', 'cod_pacto', 'cod_part', 'votos']].values

    return district_candidates.astype(int)


def groupby_sum(values, groups):
    """
    values: np.array with values to add
    groups: np.array with group indices
    return: unique group indices, sum of values by groups
    """
    unique_groups, counts = np.unique(groups, return_counts=True)
    return unique_groups, counts, np.bincount(groups, weights=values)[unique_groups]


def dhont(votes, seats, n_candidates):
    """
    :param votes: np.array, votes by group
    :param seats: int, number of seats to distribute
    :param n_candidates: np.array, number of candidates by group
    :return: np.array, seats by group
    """
    vote_matrix = np.repeat(votes[np.newaxis, :],
                            seats, axis=0)

    vote_matrix = vote_matrix / np.arange(1, seats + 1).reshape((seats, 1))

    max_seats_mask = []

    for i in range(n_candidates.shape[0]):
        mask_i = np.bincount(np.arange(n_candidates[i]), minlength=seats)[:seats]
        max_seats_mask.append(mask_i)

    max_seats_mask = np.array(max_seats_mask).T

    vote_matrix = vote_matrix * max_seats_mask
    max_indices = np.argsort(-vote_matrix.ravel())

    row_indices, col_indices = np.unravel_index(max_indices,
                                                vote_matrix.shape)

    winning_lists = col_indices[:np.sum(n_candidates)][:seats]

    return np.bincount(winning_lists, minlength=votes.shape[0])


def get_district_prelim_elect(district_candidates, seats):
    pacts, n_pact_candidates, pact_votes = groupby_sum(district_candidates[:, 4],  # candidate votes
                                                       district_candidates[:, 2])  # cod_pacto

    seats_by_pact = dhont(votes=pact_votes,
                          seats=seats,
                          n_candidates=n_pact_candidates)

    prelim_winners = []
    for i, pact in enumerate(pacts):
        pact_candidates = district_candidates[district_candidates[:, 2] == pact]
        parties, n_party_candidates, party_votes = groupby_sum(pact_candidates[:, 4],  # candidate votes
                                                               pact_candidates[:, 3])  # cod_part

        seats_by_party = dhont(votes=party_votes,
                               seats=seats_by_pact[i],
                               n_candidates=n_party_candidates)

        for j, party in enumerate(parties):
            party_candidates = pact_candidates[pact_candidates[:, 3] == party]
            sort_party_candidates = party_candidates[np.argsort(-party_candidates[:, 4])]
            winner_candidates = sort_party_candidates[:seats_by_party[j], :2]
            prelim_winners.append(winner_candidates)

    prelim_winners = np.concatenate(prelim_winners)

    return prelim_winners


def get_n_replaces(gender):
    gender_count = np.bincount(gender, minlength=2)

    n_replaces = int((gender_count[0] - gender_count[1]) / 2)

    underrepresented_gender = np.sign(np.sign(n_replaces) + 1)

    return np.abs(n_replaces), underrepresented_gender


def get_replaces(district_candidates, prelim_winners, n_replaces, underrepresented_gender):
    if n_replaces == 0:
        return [], []

    sort_candidates = district_candidates[np.argsort(-district_candidates[:, 4])]

    overrepresented_winners = sort_candidates[np.isin(sort_candidates[:, 0], prelim_winners[:, 0]) &
                                              ~np.isin(sort_candidates[:, 1], underrepresented_gender)]

    underrepresented_losers = sort_candidates[~np.isin(sort_candidates[:, 0], prelim_winners[:, 0]) &
                                              np.isin(sort_candidates[:, 1], underrepresented_gender)]

    add = []
    remove = []

    for i in range(1, overrepresented_winners.shape[0] + 1):

        party_mask = underrepresented_losers[:, 3] == overrepresented_winners[-i, 3]

        n_party_replaces = np.sum(party_mask)

        if n_party_replaces > 0:
            remove.append(overrepresented_winners[-i, 0])
            add.append(underrepresented_losers[party_mask, 0][0])

        else:
            pact_mask = underrepresented_losers[2] == overrepresented_winners[-i, 3]

            n_pact_replaces = np.sum(pact_mask)
            if n_pact_replaces > 0:
                remove.append(overrepresented_winners[-i, 0])
                add.append(underrepresented_losers[pact_mask, 0][0])

        if len(remove) == n_replaces:
            return add, remove
    return add, remove


def get_district_winners(district_candidates, seats):
    prelim_winners = get_district_prelim_elect(district_candidates, seats)

    n_replaces, underrepresented_gender = get_n_replaces(prelim_winners[:, 1])

    add, remove = get_replaces(district_candidates, prelim_winners, n_replaces, underrepresented_gender)

    return np.sort(np.concatenate([prelim_winners[:, 0][~np.isin(prelim_winners[:, 0], remove)], add]))


def get_simulated_votes(district_coms, candidate_votes_by_com, remaining_votes_by_com, n_samples, distribution):
    simulated_votes = []
    for cod in district_coms:
        # Get candidate votes in com
        com_candidate_votes = candidate_votes_by_com['votos'][candidate_votes_by_com['cod_zona'] == cod].to_numpy()

        if distribution == 'dirichlet':
            random_samples = np.random.dirichlet(com_candidate_votes + 1, n_samples)

        if distribution == 'uniform':
            random_samples = np.random.uniform(size=(com_candidate_votes.shape[0], n_samples))
            random_samples = random_samples / np.sum(random_samples, axis=0)
            random_samples = random_samples.T

        random_samples = random_samples * remaining_votes_by_com[cod]

        estimated_com_candidate_votes = com_candidate_votes + random_samples

        simulated_votes.append(estimated_com_candidate_votes)

    simulated_votes = np.sum(simulated_votes, axis=0)
    return simulated_votes.astype(int)


def get_simulated_winners(simulated_votes, seats, district_candidates):
    n_samples = simulated_votes.shape[0]

    district_candidates_ = district_candidates.copy()
    simulated_winners = []

    for i in range(n_samples):
        district_candidates_[:, 4] = simulated_votes[i]

        winners = get_district_winners(district_candidates_, seats)

        simulated_winners.append(winners)

    # Stack all simulations
    simulated_winners = np.vstack(simulated_winners).astype(int)

    return simulated_winners


def get_vote_info(votes):
    turnout = pd.read_csv('data/docs/turnout_2016_2021.csv')
    zonas_pad = load_txt('data/docs/Escenario_ZonasPadre_018032.txt')

    com_votes = votes[(votes['tipo_zona'] == 'C') & (votes['ambito'] == 7)]
    turnout = turnout.join(com_votes[['cod_zona', 'votos']].set_index('cod_zona'), on='cod_zona')
    turnout['remaining_votes'] = turnout['votos_esperados_2021'] - turnout['votos']
    turnout['estimated_remaining_votes'] = turnout.apply(lambda df: int(max(df['votos_esperados_2021'] * 0.05, df['remaining_votes'])), axis = 1)

    remaining_votes_by_com = turnout[['cod_zona', 'estimated_remaining_votes']].set_index('cod_zona').to_dict()['estimated_remaining_votes']
    candidate_votes_by_com = votes.loc[(votes['tipo_zona'] == 'C') & (votes['ambito'] == 4),
                                       ['cod_ambito', 'cod_zona', 'votos']].sort_values('cod_ambito')


    district2com = zonas_pad[(zonas_pad['tipo_zona'] == 'C') &
                             (zonas_pad['tipo_zona_pad'] == 'D')]
    district2com = dict(district2com.groupby('cod_zona_pad').apply(lambda df: df['cod_zona'].tolist()))

    return candidate_votes_by_com, district2com, candidate_votes_by_com, remaining_votes_by_com


def translate_to_imfd(simulated_winners, cand2pact, pact2imfd):
    simulated_winners_pacts = np.vectorize(cand2pact.get)(simulated_winners)

    simulated_winners_imfd = np.vectorize(pact2imfd.get)(simulated_winners_pacts)

    return simulated_winners_pacts, simulated_winners_imfd


def get_candidate_probs(simulated_winners):
    n_samples = simulated_winners.shape[0]
    candidate_probs = np.apply_along_axis(lambda x:
                                          np.bincount(x, minlength=1278),
                                          axis=1,
                                          arr=simulated_winners-800001)
    candidate_probs = np.sum(candidate_probs, axis=0) / n_samples

    return candidate_probs

