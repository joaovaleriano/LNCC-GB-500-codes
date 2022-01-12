#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

LNCC

GB-500-TEMC: Modelos Compartimentais em Epidemiologia e Inferência Bayesiana

Autor: João Pedro Valeriano Miranda

Inferência do viés de uma moeda representada por sorteios aleatórios de caras e 
coroas (0s e 1s), uniformemente distribuídas.

-------------------------------------------------------------------------------

Considere uma moeda de viés v, isto é, uma moeda que, ao ser lançada, retorna 
cara com probabilidade v, e coroa com probabilidade 1-v.

Dado um viés v, a probabilidade de esta moeda retornar y caras em N jogadas é 
dada pela distribuição binomial de taxa de sucesso v e número de tentativas N. 
Esta é a likelihood:
    
    P(y|v, N) = N! / ( y! * (N-y)! ) * v^y * (1-v)^(N-y).
    
Se não temos nenhuma informação sobre o viés da moeda, utilizamos uma prior
uniforme:
    
    P(v) = U(0, 1).
    
Considerando a likelihood margianl P(y) como apenas uma normalização, a menos
disto, temos a distribuição posterior:
    
    P(v|y, N) = N! / ( y! * (N-y)! ) * v^y * (1-v)^(N-y), 0 <= v <= 1.
    
Vamos sortear várias jogadas da moeda, como 0s e 1s uniformemente distribuídos,
e e gerar a distribuição posterior a partir dos resultados.

* Cara == 0
  Coroa == 1

"""

import numpy as np # arrays etc
from math import factorial as fact # fatorial
import matplotlib.pyplot as plt # gráficos
from scipy.integrate import simps # método de Simpson para integral
from scipy.special import binom # binômio de Newton

# Fixando seed dos números aleatórios
np.random.seed(123456789)

# PDF da distribuição normal, para uso como aproximação da distribuiçãobinomial, 
# no caso de grande número de tentativas.
def normal_pdf(v, y, N):
    
    return np.exp(-(y-N*v)**2/(N*v*(1-v))/2)/np.sqrt(2*np.pi*N*v*(1-v))

# PDF da distribuição binomial
def binom_pdf(v, y, N):
    
    # Se N e y forem grandes demais para o cálculo numérico do binômio de Newton, 
    # aproximamos a PDF por uma gaussiana, como é sabido pelo Teorema de 
    # De Moivre-Laplace
    if binom(N, y) != np.inf:
        return binom(N, y)*v**y*(1-v)**(N-y)
    
    else:
        print(f"** PDF para {N} jogadas gerada através de aproximação pela distribuição normal. **")
        return normal_pdf(v, y, N)

v = np.linspace(0, 1, 1000)[1:-1] # Intervalo em que consideramos o possível viés da moeda

N = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000] # Números de jogadas para os quais vamos plotar a posterior

fig, ax = plt.subplots(4, 3, figsize=(20,15), sharex=True) # Criando figura
fig.text(0.07, 0.5, "Distribuição posterior do viés da moeda", fontsize=24, va="center", rotation="vertical")
fig.text(0.5, 0.07, "Viés da moeda", fontsize=24, ha="center")

caras = 0 # Número inicial de caras

# loop adicionando jogadas da moeda
for n in range(0, N[-1]+1):
    if n in N: # se o número de jogadas atual deve ser plotado, o fazemos
        pdf = binom_pdf(v, caras, n) # PDF da dist. binomial
        
        # normalização da distribuição
        norm = simps(pdf, v)
        pdf /= norm
        
        # Plotagem
        plt.subplot(4, 3, N.index(n)+1)
        plt.plot(v, pdf, lw=3, label=f"{n} jogadas")
        plt.plot([], [], " ", label=f"{caras} caras")
        plt.fill_between(v, pdf, alpha=0.5)
        plt.vlines(0.5, 0, np.max(pdf), "k", "--", lw=2)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlim(0, 1)
        plt.ylim(0)
        plt.grid()
        plt.legend(loc="upper right", fontsize=14)
    
    # n-ésima jogada
    caras += 1 - np.random.randint(0, 2)
    
# mostrar figura
# plt.show()

# ou salvar figura
plt.savefig("vies_moeda_posterior.png", dpi=300, bbox_inches="tight")
plt.close()