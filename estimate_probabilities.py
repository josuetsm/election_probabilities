import glob

import numpy as np

from utils import load_txt
from election_winners import *

files = sorted(glob.glob('data/raw/*.zip'))
candidates = load_txt('data/docs/Escenario_Candidatos_018032.txt')

for file in files[:3]:
    votes = load_txt(file)
    print(file, votes.shape)

district_candidates = get_district_candidates(district=6028,
                                              candidates=candidates,
                                              votes=votes)

district_prelim_elect = get_district_prelim_elect(district_candidates, 3)
district_prelim_elect

get_n_replaces(district_prelim_elect[:,1])