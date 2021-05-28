import glob
from utils import load_txt

files = sorted(glob.glob('data/raw/*.zip'))

for file in files[:3]:
    votes = load_txt(file)
    print(file, votes.shape)


candidates = load_txt('data/docs/Escenario_Candidatos_018032.txt')
elections = load_txt('data/docs/Escenario_Elecciones_018032.txt')
pacts = load_txt('data/docs/Escenario_Pactos_018032.txt')
parties = load_txt('data/docs/Escenario_Partidos_018032.txt')
sub_pacts = load_txt('data/docs/Escenario_SubPactos_018032.txt')
zones = load_txt('data/docs/Escenario_Zonas_018032.txt')
father_zones = load_txt('data/docs/Escenario_ZonasPadre_018032.txt')


