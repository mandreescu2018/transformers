import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import textwrap


def plot_cm(cm):
    classes = ['negative', 'positive']
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    ax = sn.heatmap(df_cm, annot=True, fmt='g')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    plt.show()

def wrap(x):
    return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)

