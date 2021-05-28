import os
import glob

import matplotlib.pyplot as plt

from election_process import *
from utils import get_candidates_info, get_time


data_path = 'data/results'

real_winners_counts = np.load(os.path.join(data_path, 'real_winners_counts.npy'))
simulated_winners_counts = np.load(os.path.join(data_path, 'simulated_winners_counts.npy'))

files = sorted(glob.glob('data/raw/*.zip'))
candidates, cand2pact, pact2imfd, cod2glosa_imfd = get_candidates_info()

prop_votes = [int(file.split('_')[-2]) / 10000 for file in files[4:]]
time = [file.split('_')[-1][:-8] for file in files[4:]]
time = [t[:2] + ':' + t[2:] for t in time]

# Confidence interval
imfd_ci = []

for p in range(11):
    imfd_ci.append(np.apply_along_axis(get_ci, axis = 1, arr=simulated_winners_counts[:, :, p]))
imfd_ci = np.array(imfd_ci)


fig, axs = plt.subplots(3, 2)
fig.tight_layout(h_pad=1.5)
axs = axs.ravel()
for i, p in enumerate([0, 1, 2, 8, 9, 10]):
    axs[i].plot(prop_votes, imfd_ci[p, :, 0], '--', color=f'C{i}')
    axs[i].fill_between(prop_votes, imfd_ci[p, :, 1], imfd_ci[p, :, 2], color=f'C{p}', alpha=.1)
    axs[i].plot(prop_votes, len(prop_votes) * [real_winners_counts[-1, p]], '-', color=f'C{i}')
    axs[i].plot(prop_votes, real_winners_counts[:, p], 'o', color=f'C{i}')
    axs[i].set_title(cod2glosa_imfd[p+1], size=10, loc = 'left')
    axs[i].set_xlabel('Mesas escrutadas')
    axs[i].set_ylabel('Escaños')
    ax_mid = np.round(np.mean(axs[i].dataLim.get_points()[:, 1]))
    axs[i].set_yticks(np.arange(ax_mid - 5, ax_mid + 6))

plt.legend(['Media simulaciones', 'Resultado final', 'Resultados parciales'])
plt.show()


time = get_time()

fig, axs = plt.subplots(3, 2)
fig.tight_layout(h_pad=1.5)
axs = axs.ravel()
for i, p in enumerate([0, 1, 2, 8, 9, 10]):
    axs[i].plot(time['atime'], imfd_ci[p, :, 0], '--', color=f'C{i}')
    axs[i].fill_between(time['atime'], imfd_ci[p, :, 1], imfd_ci[p, :, 2], color=f'C{p}', alpha=.1)
    axs[i].plot(time['atime'], len(prop_votes) * [real_winners_counts[-1, p]], '-', color=f'C{i}')
    axs[i].plot(time['atime'], real_winners_counts[:, p], 'o', color=f'C{i}')
    axs[i].set_title(cod2glosa_imfd[p+1], size=10, loc = 'left')
    axs[i].set_xlabel('Hora')
    axs[i].set_ylabel('Escaños')
    ax_mid = np.round(np.mean(axs[i].dataLim.get_points()[:, 1]))
    axs[i].set_yticks(np.arange(ax_mid - 5, ax_mid + 6))
    axs[i].set_xticks(time['atime'])
    axs[i].set_xticklabels(time['atime_labels'], rotation=30, fontdict =  {'fontsize': 7})

plt.legend(['Media simulaciones', 'Resultado final', 'Resultados parciales'])
plt.show()



candidate_probs = get_candidate_probs(simulated_winners_f)
estimate_winners = np.arange(800001, 801279)[np.argsort(-candidate_probs)][:138]
simulated_winners_pacts, simulated_winners_imfd = translate_to_imfd(simulated_winners_f, cand2pact, pact2imfd)





predict.append(len(np.intersect1d(winners, estimate_winners)))
print(predict)

prop_votes = [int(file.split('_')[-2]) / 10000 for file in files]

plt.plot(prop_votes, predict, 'o-')
plt.xlabel('Mesas escrutadas')
plt.ylabel('Candidaturas predichas')
plt.ylim(0, 145)
plt.hlines(138, xmin=0, xmax=1, color='red')
plt.show()

for i in range(len(prop_votes)):
    plt.annotate(str(prop_votes[i]) + '%',
                 xy=(prop_votes[i], predict[i]),
                 xytext=(prop_votes[i], predict[i]),
                 fontsize=10)

for file in files:
    votes = load_txt(file)
    candidate_votes_by_com, district2com, candidate_votes_by_com, remaining_votes_by_com = get_vote_info(votes)
    print(file, votes.shape,
          votes[votes['tipo_zona'] == 'G'].votos.sum(),
          sum(remaining_votes_by_com.values()))

# Para cada momento
# dos momentos, inicio, intermedio y final
# Está la foto actual, está la predicción, y está la comparación de la predicción con el final

# Visualización por distrito


# 1) Datos de listas
# 2)
# 3) Recall y precision por distrito y nacional
# Ver qué distrito era más complejo

# 0.8


import numpy as np
import scipy.stats




simulated_winners_imfd.shape

simulated_winners_imfd
