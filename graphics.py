#Import
import numpy as np 
import math
import matplotlib.pyplot as plt
def eng_plot(pot_eng, kin_eng, tot_eng):
    plt.plot(pot_eng, color = 'm')
    plt.plot(kin_eng, color = 'y')
    plt.plot(tot_eng, color = 'g')
    plt.xlabel("Zeitschritt")
    plt.ylabel("Energie")
    plt.legend()
    plt.show()

def temp_plot (inst_temp, temp_average):
    plt.plot(inst_temp, color = 'r')
    plt.title("Durschnittstempertur: " + str(temp_average))
    plt.xlabel("Zeitschritt")
    plt.ylabel("Temperatur")
    plt.legend()
    plt.show()

def temp_histogramm (temp_distribution):
    plt.figure(figsize=(8, 5))
    bins=temp_distribution[0]
    counts = temp_distribution[1]
    breite = bins[1] - bins[0]
    plt.bar(bins, counts, width=breite, align='edge', color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Histogramm der Observablen (Bins: 25)")
    plt.xlabel("Wert der Observonablen")
    plt.ylabel("Häufigkeit")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()