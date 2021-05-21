import numpy as np


def recode_party(df):
    return df['cod_part'] - 99 + df['can_orden'] if df['cod_part'] == 999099 else df['cod_part']


def recode_pact(df):
    return df['cod_pacto'] - 99 + df['can_orden'] if df['cod_part'] == 999 else df['cod_pacto']


def get_district_candidates(district, candidates, votes):
    """
    district: int district code 6001-6028
    candidates: pd.DataFrame candidates
    votes: pd.DataFrame with votes
    return: np.array of shape = (5, n) (n: number of district candidates)
    """
    candidates = candidates.copy()

    candidates['cod_part'] = candidates.apply(recode_party, axis=1)
    candidates['cod_pacto'] = candidates.apply(recode_pact, axis=1)

    distric_votes = votes[votes['cod_zona'] == district].set_index('cod_ambito')
    distric_votes = distric_votes[['votos']]

    district_candidates = candidates.join(distric_votes, on='cod_cand', how='inner')
    district_candidates['cod_genero_int'] = district_candidates['cod_genero'].replace({'F': 0, 'M': 1})
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
    votes: np.array with total votes by PACTO or PARTIDO
    seats: int number of seats to distribute
    n_candidates: np.array con cantidad de candidatos por lista o partido (ordenados por codigo)
    return: np.array con número de escaños designados para cada lista o partido (ordenados por codigo)
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

    sub_gender = np.sign(np.sign(n_replaces) + 1)

    return np.abs(n_replaces), sub_gender


def fix_parity(candidatos_clean, id_electos_preliminares, reemplazos_necesarios, genero_subr):
    if reemplazos_necesarios == 0:
        return id_electos_preliminares, [], []
    # Ordenar candidatos por votos
    votos_sort = np.argsort(-candidatos_clean[4])
    candidatos_sort = candidatos_clean[:, votos_sort]
    # Dividir electos sobrerepresentados y no electos subrepresentados
    electos_mask = np.isin(candidatos_sort[0], id_electos_preliminares)
    subrepresentados_mask = np.isin(candidatos_sort[1], genero_subr)
    # Electos sobrerepresentados
    electos_sobrerepresentados = candidatos_sort[:, electos_mask & ~subrepresentados_mask]
    # No electos subrepresentados
    no_electos_subrepresentados = candidatos_sort[:, ~electos_mask & subrepresentados_mask]
    # Recorrer electos sobre representados de abajo hacia arriba buscando reemplazos
    ids_eliminados = []
    ids_agregados = []
    for i in range(1, electos_sobrerepresentados.shape[1] + 1):
        # Mask de partidos del candidato a reemplazar
        mask = np.where(no_electos_subrepresentados[3] == electos_sobrerepresentados[3, -i], True, False)
        # Cantidad de reemplazos disponibles en su partido
        n_reemplazos_partido = np.sum(mask)
        if n_reemplazos_partido > 0:
            # Reemplazar en partido
            ids_eliminados.append(electos_sobrerepresentados[0, -i])
            ids_agregados.append(no_electos_subrepresentados[0, mask][0])
        else:
            # Mask de listas del candidato a reemplazar
            mask = np.where(no_electos_subrepresentados[2] == electos_sobrerepresentados[2, -i], True, False)
            # Cantidad de reemplazos disponibles en su pacto
            n_reemplazos_pacto = np.sum(mask)
            if n_reemplazos_pacto > 0:
                # Reemplazar en pacto
                ids_eliminados.append(electos_sobrerepresentados[0, -i])
                ids_agregados.append(no_electos_subrepresentados[0, mask][0])
        # Reemplazos necesarios completados
        if len(ids_eliminados) == reemplazos_necesarios:
            return np.sort(np.concatenate(
                [np.setdiff1d(id_electos_preliminares,
                              ids_eliminados),
                 ids_agregados]
            )).astype(np.int64), ids_agregados, ids_eliminados
    # Caso en que no se logra obtener paridad completa
    return np.sort(np.concatenate([np.setdiff1d(id_electos_preliminares, ids_eliminados),
                                   ids_agregados])).astype(np.int64), ids_agregados, ids_eliminados
