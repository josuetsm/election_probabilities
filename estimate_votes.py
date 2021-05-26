import pandas as pd
import matplotlib.pyplot as plt

turnout = pd.read_csv('data/docs/turnout_2016_2020.csv')

turnout['votos_esperados_2021'] = turnout[['part_2016',
                                           'part_2017',
                                           'part_2020']].mean(axis=1) * turnout['electores_2021']

turnout['votos_esperados_2021'] = turnout['votos_esperados_2021'].apply(int)

for i, label in enumerate(['votos_2016',
                           'votos_2017',
                           'votos_2020']):
    x = turnout['votos_esperados_2021']
    y = turnout[label]
    plt.plot(x, y, '.', color=f'C{i}')

plt.plot([0, 200000], [0, 200000])
plt.xlabel('Votos estimados 2021')
plt.ylabel('Votos')
plt.legend([2016, 2017, 2020])

turnout.to_csv('data/docs/turnout_2016_2021.csv', index=False)