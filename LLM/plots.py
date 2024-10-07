import matplotlib.pyplot as plt
import pandas as pd
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
    limiteMenor = df['error'].min()-0.001
    limiteMayor = df['error'].max()+0.001
    return limiteMenor, limiteMayor

def filterByWord(df, word_id):
    if(word_id != ''):
        df = df[(df['wordID'] == word_id)]
    return df

def get_plot_with_scatterplot(df, basepath, titulo_segun, model, size = (10,40), target_a_graficar = 'todos los targets', name = 'allTargets', word_id = ''):
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
                    label=f'Target {target} con meaning ID {meaning_id}',
                    color=colors_extended[i])
    plt.errorbar(layers, meanErrors, yerr=standardErrors, fmt='o', color='black', capsize=5, label='Error estándar')
    plt.title(f"Scatter plot de sesgos para {target_a_graficar} {titulo_segun}en cada capa de {model}")
    plt.xlabel("Capa del modelo")
    plt.ylabel("Diferencia entre sesgo generado y sesgo base")
    plt.xticks(layers)
    plt.ylim(limiteMenor, limiteMayor)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{basepath}pltLayers_{name}_scatter.png')
    plt.close()

def get_plots_for_each_target(df, basepath, titulo_segun, model):
    unique_combinations = df[['wordID']].drop_duplicates()
    for word_id in unique_combinations.values:
        word_id = word_id[0]
        subset_word = df[(df['wordID'] == word_id)]
        target = subset_word['target'].unique()
        get_plot_with_scatterplot(df, basepath, titulo_segun, model, (10,15), f'el target {target}', f'target_{int(word_id)+1}', word_id)

def openCsvAsDf(listCsv):
    listDf = []
    for csv in listCsv:
        listDf.append(pd.read_csv(csv,quotechar='"'))
    return listDf

def get_plot_with_more_than_one_csv(listCsv, basepath, titulo_segun, labels, labelY):
    listDf = openCsvAsDf(listCsv)    
    colors_extended = getColors(len(listCsv))
    #plt.figure(figsize=(10,15))
    for i,df in enumerate(listDf):
        layers, meanErrors, standardErrors = getLayersMeanAndStandardError(df)
        plt.errorbar(layers, meanErrors, yerr=(standardErrors), 
                    label=labels[i],
                    color=colors_extended[i])
    plt.title(f"Comparison of biases {titulo_segun}\n in each layer of the models")
    plt.xlabel("Layers")
    plt.ylabel(labelY)
    plt.legend(loc='best')
    #plt.ylim(-0.02, 0.035)
    plt.savefig(f'{basepath}plot-layers.png')
    plt.savefig(f'{basepath}plot-layers.svg')
    plt.close()

def getPlots(lista_de_df, layers, basepath, titulo_segun, m, tipoMetrica):
    df = get_errores_para_todas_las_capas(lista_de_df, layers, basepath, tipoMetrica)
    ## Para graficar una linea con errores estandar
    get_plot(df, f'{basepath}plots/', titulo_segun, m)
    ## Para graficar con scatterplot
    get_plot_with_scatterplot(df, f'{basepath}plots/', titulo_segun, m)
    get_plots_for_each_target(df, f'{basepath}plots/', titulo_segun, m)