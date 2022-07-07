import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import array

STEPS = 100


def draw_distr_plots(init_data):
    for column in init_data.columns:
        plt.title(label=column)
        col_df = init_data[column]
        col_values = col_df.dropna(how='any', axis=0).values
        sns.kdeplot(col_values, shade=True)

        plt.axvline(x=col_df.quantile(0.25, interpolation='linear'))
        plt.axvline(x=col_df.quantile(0.50, interpolation='linear'))
        plt.axvline(x=col_df.quantile(0.75, interpolation='linear'))
        plt.show()


def draw_heatmap(init_data):
    corr = init_data.corr()
    plt.figure(figsize=(12, 12))
    sns.heatmap(abs(corr), annot=True, fmt='.1g', cmap='coolwarm')
    plt.show()


def draw_gains(init_data, target_attr):
    g_values = init_data[target_attr].values
    g_class_marks = array([None] * len(g_values))
    g_min = min(g_values)
    g_max = max(g_values)
    g_step = (g_max - g_min) / STEPS

    for g in range(len(g_values)):
        for i in range(STEPS + 1):
            if g_min + i * g_step <= g_values[g] < g_min + (i + 1) * g_step:
                g_class_marks[g] = i

    g_marks = Counter(g_class_marks)

    # Значение info(T)
    infoT = 0.0
    for i in range(STEPS + 1):
        frac = g_marks[i] / len(g_values)
        if frac > 0:
            infoT -= frac * math.log2(frac)

    gains = []
    for i in range(len(init_data.columns) - 2):
        colna = init_data[init_data.columns.values[i]]
        col = colna.dropna(how='any', axis=0)

        Tis = Counter(col)
        infox = 0.0
        splitx = 0.0

        for Ti in Tis:
            marks = []
            for l in range(row_count):
                if colna[l] == Ti:
                    marks.append(g_class_marks[l])
            freq = Counter(marks)

            # Значение info_x(Ti)
            for k in range(STEPS + 1):
                frac = freq[k] / len(marks)
                if frac > 0:
                    infox -= len(marks) / len(g_values) * frac * math.log2(frac)
                    
            frac = len(marks) / len(g_values)
            splitx -= frac * math.log2(frac)
        gain_ratio = 0
        if splitx > 0:
            gain_ratio = (infoT - infox) / splitx
        # print(colna.name + ":")
        # print("(" + str(infoT) + " - " + str(infox) + ") / " + str(splitx))
        # print(gain_ratio)
        gains.append(gain_ratio)
    plt.plot(columns[:29], gains)
    plt.show()


if __name__ == '__main__':
    initial_data = pd.read_csv('Data.csv', sep=';')
    initial_data.info()
    columns = initial_data.columns.values
    row_count = initial_data.shape[0]

    # Тепловая карта
    draw_heatmap(initial_data)
    
    # Графики распределения с 1 и 3 квартилями
    draw_distr_plots(initial_data)

    # Гистограммы важности признаков по gain_ratio для целевых переменных
    draw_gains(initial_data, 'G_total')
    draw_gains(initial_data, 'КГФ')
