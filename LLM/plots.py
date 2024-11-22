import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from getErrors import get_errores_para_todas_las_capas

## Para las metricas

def getLayersMeanAndStandardError(df):
    generalError = df[(df['wordID'] == (df['wordID'].iloc[0])) & (df['meaningID'] == (df['meaningID'].iloc[0]))]
    layers = generalError['layer']
    meanErrors = generalError['meanError']
    standardErrors = generalError['standardError']
    return layers, meanErrors, standardErrors

def get_plot(df, basepath, titulo_segun, model):
    layers, meanErrors, standardErrors = getLayersMeanAndStandardError(df)
    plt.figure(figsize=(10,15))
    plt.errorbar(layers, meanErrors, yerr=(standardErrors))
    plt.title(f"Sesgos {titulo_segun}en cada capa de {model}")
    plt.xlabel("Capas")
    plt.ylabel("Diferencia entre sesgo generado y sesgo base")
    plt.ylim(-0.02, 0.035)
    plt.savefig(f'{basepath}plot-layers.png')
    plt.savefig(f'{basepath}plot-layers.svg')
    plt.close()

def getColors(num_colors):
    colors = plt.cm.tab20.colors
    return colors * (num_colors // len(colors)) + colors[:num_colors % len(colors)]

def getLimits(df):
    limiteMenor = df['error'].min()-0.005
    limiteMayor = df['error'].max()+0.005
    return limiteMenor, limiteMayor

def filterByWord(df, word_id):
    if(word_id != ''):
        df = df[(df['wordID'] == word_id)]
    return df

def get_plot_with_scatterplot(df, basepath, titulo_segun, model, size = (8,5), target_a_graficar = 'todos los targets', name = 'allTargets', word_id = ''):
    TITLESIZE = 10
    layers, meanErrors, standardErrors = getLayersMeanAndStandardError(df)
    colors_extended = getColors(len(df[df['layer'] == 0]))
    plt.figure(figsize=size)
    limiteMenor, limiteMayor = getLimits(df)
    df = filterByWord(df, word_id)
    unique_combinations = df[['wordID', 'meaningID']].drop_duplicates()
    for i, (word_id, meaning_id) in enumerate(unique_combinations.values):
        subset = df[(df['wordID'] == word_id) & (df['meaningID'] == meaning_id)]
        target = subset['target'].to_list()[0]
        plt.scatter(subset['layer'], subset['error'], 
                    label=f'Target "{target}" con significado {meaning_id}',
                    color=colors_extended[i])
    plt.errorbar(layers, meanErrors, yerr=standardErrors, fmt='o', color='black', capsize=TITLESIZE, label='Sesgo promedio con error estándar')
    #plt.title(f"Scatter plot de sesgos para {target_a_graficar} {titulo_segun}en cada capa de {model}")
    plt.xlabel("Capas", fontsize=TITLESIZE)
    plt.ylabel("Diferencia entre sesgo generado y sesgo base", fontsize=TITLESIZE)
    plt.xticks(layers)
    plt.ylim(limiteMenor, limiteMayor)
    plt.legend(fontsize=TITLESIZE)
    plt.grid(True)
    plt.savefig(f'{basepath}pltLayers_{name}_scatter.png')
    plt.close()

def get_plots_for_each_target(df, basepath, titulo_segun, model):
    unique_combinations = df[['wordID']].drop_duplicates()
    for word_id in unique_combinations.values:
        word_id = word_id[0]
        subset_word = df[(df['wordID'] == word_id)]
        target = subset_word['target'].unique()
        get_plot_with_scatterplot(df, basepath, titulo_segun, model, (8,5), f'el target {target}', f'target_{int(word_id)+1}', word_id)

def openCsvAsDf(listCsv):
    listDf = []
    for csv in listCsv:
        listDf.append(pd.read_csv(csv,quotechar='"'))
    return listDf

def get_plot_with_more_than_one_csv(listCsv, basepath, titulo_segun, labels, labelY):
    LABELSIZE = 8 
    TITLESIZE = 8
    listDf = openCsvAsDf(listCsv)    
    plt.figure(figsize=(5,3))
    for i,df in enumerate(listDf):
        label = labels[i]
        if "Word Level" in label:
            color = "blue"
        else:
            color = "red"
        if "categoria" in label:
            linestyle = "--"
        else:
            linestyle = "-"
        layers, meanErrors, standardErrors = getLayersMeanAndStandardError(df)
        plt.errorbar(layers, meanErrors, yerr=(standardErrors), 
                    label=label, color=color, linestyle=linestyle)
    #plt.title(f"Comparison of biases {titulo_segun}\n in each layer of the models")
    plt.xlabel("Capas", fontsize=TITLESIZE)
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
    plt.ylabel(labelY, fontsize=TITLESIZE)
    plt.legend(loc='upper left', fontsize=TITLESIZE)
    plt.tick_params(axis='both', which='major', labelsize=LABELSIZE)
    #plt.ylim(-0.02, 0.05)
    plt.tight_layout()
    plt.savefig(f'{basepath}plot-layers.png')
    plt.savefig(f'{basepath}plot-layers.svg')
    plt.close()

def get_plot_with_different_tokenizer(listCsv, basepath, titulo_segun, labels, labelY):
    LABELSIZE = 8 
    TITLESIZE = 8
    listDf = openCsvAsDf(listCsv)    
    plt.figure(figsize=(6,4))    
    for i,df in enumerate(listDf):
        label = labels[i]
        if "primer" in label:
            color = "blue"
        else:
            color = "red"
        if "categoría" in label:
            linestyle = "--"
        else:
            linestyle = "-"
        layers, meanErrors, standardErrors = getLayersMeanAndStandardError(df)
        plt.errorbar(layers, meanErrors, yerr=(standardErrors), 
                    label=label, color=color, linestyle=linestyle)
    #plt.title(f"Comparison of biases {titulo_segun}\n in each layer of the models")
    plt.xlabel("Capas", fontsize=TITLESIZE)
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
    plt.ylabel(labelY, fontsize=TITLESIZE)
    plt.legend(loc='upper left', fontsize=TITLESIZE)
    plt.tick_params(axis='both', which='major', labelsize=LABELSIZE)
    plt.ylim(-0.0015, 0.035)
    plt.tight_layout()
    plt.savefig(f'{basepath}plot-layers.png')
    plt.savefig(f'{basepath}plot-layers.svg')
    plt.close()

def get_plot_with_different_cluster(listCsv, basepath, titulo_segun, labels, labelY, model):
    LABELSIZE = 8 
    TITLESIZE = 8
    listDf = openCsvAsDf(listCsv)    
    plt.figure(figsize=(5,3))
    for i,df in enumerate(listDf):
        label = labels[i]
        if "Baja" in label:
            color = sns.color_palette('tab10', 10)[2]
        else:
            if "Alta" in label:
                color = sns.color_palette('tab10', 10)[1]
            else:
                color = sns.color_palette('tab10', 10)[0]
        layers, meanErrors, standardErrors = getLayersMeanAndStandardError(df)
        plt.errorbar(layers, meanErrors, yerr=(standardErrors), 
                    label=label, color=color)
    #plt.title(f"Comparacion de sesgo {titulo_segun}\n en cada capa del modelo")
    plt.xlabel("Capas", fontsize=TITLESIZE)
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
    plt.ylabel(labelY, fontsize=TITLESIZE)
    plt.legend(loc='upper left', fontsize=TITLESIZE)
    plt.tick_params(axis='both', which='major', labelsize=LABELSIZE)
    plt.ylim(-0.0015, 0.045)
    plt.tight_layout()
    plt.savefig(f'{basepath}plot-layers-{model}.png')
    plt.savefig(f'{basepath}plot-layers-{model}.svg')
    plt.close()

def getPlots(lista_de_df, layers, basepath, titulo_segun, m, tipoMetrica):
    df = get_errores_para_todas_las_capas(lista_de_df, layers, basepath, tipoMetrica)
    ## Para graficar una linea con errores estandar
    get_plot(df, f'{basepath}plots/', titulo_segun, m)
    ## Para graficar con scatterplot
    get_plot_with_scatterplot(df, f'{basepath}plots/', titulo_segun, m)
    get_plots_for_each_target(df, f'{basepath}plots/', titulo_segun, m)