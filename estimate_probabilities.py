import glob

import numpy as np
import pandas as pd

from utils import load_txt
from settings import district_seats
from election_process import *

import matplotlib.pyplot as plt



files = sorted(glob.glob('data/raw/*.zip'))
candidates = load_txt('data/docs/Escenario_Candidatos_018032.txt')
candidates = candidates[candidates['cod_elec'] == 8]


servel_imfd = pd.read_csv('data/docs/pactos_servel_imfd.csv')
cand2pact = dict(zip(candidates['cod_cand'], candidates['cod_pacto']))

pact2imfd = dict(zip(servel_imfd['COD_PACTO'], servel_imfd['COD_IMFD']))
pact2imfd[175] = 10 # D26
pact2imfd[176] = 10 # D28
cod2glosa_imfd = dict(zip(servel_imfd['COD_IMFD'], servel_imfd['GLOSA_IMFD']))

winners = []
for file in files[-1:]:
    votes = load_txt(file)
    candidate_votes_by_com, district2com, candidate_votes_by_com, remaining_votes_by_com = get_vote_info(votes)
    print(file, votes.shape,
          votes[votes['tipo_zona'] == 'G'].votos.sum(),
          sum(remaining_votes_by_com.values()))

    for i, k in tqdm(enumerate(range(6001, 6029))):
        district_candidates = get_district_candidates(district=k,
                                                      candidates=candidates,
                                                      votes=votes)

        district_winners = get_district_winners(district_candidates, district_seats[i])
        winners.append(district_winners)
    winners = np.concatenate(winners).astype(int)



predict = []
for file in files:
    votes = load_txt(file)
    candidate_votes_by_com, district2com, candidate_votes_by_com, remaining_votes_by_com = get_vote_info(votes)
    print(file, votes.shape,
          votes[votes['tipo_zona'] == 'G'].votos.sum(),
          sum(remaining_votes_by_com.values()))

    n_samples = 100
    simulated_winners = np.array([], dtype = int).reshape(n_samples, 0)

    for i, k in tqdm(enumerate(range(6001, 6029))):
        district_candidates = get_district_candidates(district=k,
                                                      candidates=candidates,
                                                      votes=votes)

        #district_winners = get_district_winners(district_candidates, district_seats[i])
        #winners.append(district_winners)

        district_coms = district2com[k]
        simulated_votes = get_simulated_votes(district_coms,
                                              candidate_votes_by_com,
                                              remaining_votes_by_com,
                                              n_samples=n_samples,
                                              distribution='dirichlet')

        district_simulated_winners = get_simulated_winners(simulated_votes, district_seats[i], district_candidates)

        simulated_winners = np.hstack([simulated_winners, district_simulated_winners])


    simulated_winners_pacts, simulated_winners_imfd = translate_to_imfd(simulated_winners, cand2pact, pact2imfd)

    candidate_probs = get_candidate_probs(simulated_winners)
    estimate_winners = np.arange(800001, 801279)[np.argsort(-candidate_probs)][:138]

    predict.append(len(np.intersect1d(winners, estimate_winners)))
    print(predict)

#plt.hist(candidate_probs, bins=np.linspace(0, 1, 100), alpha = 0.6)
#plt.legend([f"{int(file.split('_')[-2])/100}%" for file in files])


mesas_escrutadas = [int(file.split('_')[-2])/10000 for file in files]

#plt.figure(figsize=(20, 10))
plt.plot(mesas_escrutadas, predict, 'o-')
plt.xlabel('Mesas escrutadas')
plt.ylabel('Candidaturas predichas')
plt.ylim(0, 145)
plt.hlines(138, xmin=0, xmax=1, color = 'red')
plt.show()
for i in range(len(mesas_escrutadas)):
    plt.annotate(str(mesas_escrutadas[i])+'%',
                 xy=(mesas_escrutadas[i], predict[i]),
                 xytext=(mesas_escrutadas[i], predict[i]),
                 fontsize = 10)
