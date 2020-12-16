# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:32:16 2020

@author: wende
"""
import numpy as np
import matplotlib.pyplot as plt

def salvar_grafico(metricas, int_conf):
    
    for m in metricas.columns:
        fig, bx = plt.subplots()
        width=0.5
        clf = ['RegLogística', 'RandomForest', 'MLP', 'SVM', 'NaiveBayes', 'Dummy']
        plt.title("{} (%)".format(m))
        barra = bx.barh(clf, metricas.loc[:, m], width, xerr=int_conf.loc[:, m])
        plt.gca().invert_yaxis()
        for i, v in enumerate(metricas.loc[:, m]):
            if m == 'FPR':
                bx.xaxis.set_ticks(np.arange(0, 85, 5)) #sem clus
                #bx.xaxis.set_ticks(np.arange(0, 60, 5)) #com clus
                plt.text(v + 3, i + 0.1, str(v))
            elif m == 'FNR':
                bx.xaxis.set_ticks(np.arange(0, 50, 5)) #sem clus
                #bx.xaxis.set_ticks(np.arange(0, 75, 5)) #com clus
                plt.text(v + 3.5, i + 0.1, str(v))
            else:
                bx.xaxis.set_ticks(np.arange(0, 110, 10))
                plt.text(v - 12.5, i + 0.1, str(v), color="white")
        plt.savefig(r'C:\Users\wende\OneDrive\Imagens\Imagens Warley\metricas\sem clus\{}.png'.format(m), bbox_inches="tight", dpi=150)


def gerar_csv(dataframe_metrica):
    colunas = ['RegLogística', 'RandomForest', 'MLP', 'SVM', 'NaiveBayes', 'Dummy']
    metricas = ['Acurácia', 'F1', 'G-mean', 'Precisão', 'FPR', 'FNR', 'Sensibilidade', 'Especificidade']
    for m in metricas:
        dataframe_metrica.columns = colunas
        dataframe_metrica.to_csv(r'C:\Users\wende\Videos\Curso Warley\Avaliação\Dados{}.csv'.format(m), index=False, header=True)