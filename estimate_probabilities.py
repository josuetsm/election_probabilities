import os
import glob

from election_process import *
from settings import district_seats

from utils import get_candidates_info

files = sorted(glob.glob('data/raw/*.zip'))

candidates, cand2pact, pact2imfd, cod2glosa_imfd = get_candidates_info()

real_winners_ = []
simulated_winners_ = []

for file in files[4:]:
    votes = load_txt(file)
    candidate_votes_by_com, district2com, remaining_votes_by_com = get_vote_info(votes)
    print('File:', file)
    print('Votes:', votes[votes['tipo_zona'] == 'G'].votos.sum())

    n_samples = 1000
    simulated_winners_f = np.array([], dtype=int).reshape(n_samples, 0)
    real_winners_f = []
    for i, k in tqdm(enumerate(range(6001, 6029))):
        district_candidates = get_district_candidates(district=k,
                                                      candidates=candidates,
                                                      votes=votes)

        district_winners = get_district_winners(district_candidates, district_seats[i])

        real_winners_f.append(district_winners)

        district_coms = district2com[k]
        simulated_votes = get_simulated_votes(district_coms,
                                              candidate_votes_by_com,
                                              remaining_votes_by_com,
                                              n_samples=n_samples,
                                              distribution='dirichlet')

        district_simulated_winners = get_simulated_winners(simulated_votes, district_seats[i], district_candidates)

        simulated_winners_f = np.hstack([simulated_winners_f, district_simulated_winners])

    real_winners_f = np.concatenate(real_winners_f)
    real_winners_.append(real_winners_f)
    simulated_winners_.append(simulated_winners_f)

# Stack
real_winners = np.array(real_winners_)
simulated_winners = np.array(simulated_winners_)

# Translate to imfd
real_winners_pacts, real_winners_imfd = translate_to_imfd(real_winners, cand2pact, pact2imfd)
simulated_winners_pacts, simulated_winners_imfd = translate_to_imfd(simulated_winners, cand2pact, pact2imfd)

# Counts
real_winners_counts = get_pact_imfd_counts(real_winners_imfd)
simulated_winners_counts = get_pact_imfd_counts(simulated_winners_imfd)

save_path = 'data/results'
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.save(os.path.join(save_path, 'real_winners_counts.npy'), real_winners_counts)
np.save(os.path.join(save_path, 'simulated_winners_counts.npy'), simulated_winners_counts)