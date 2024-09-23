import matplotlib.pyplot as plt
import pandas as pd

## Para las metricas

def getCombinedDFAndExport(lista_df, basepath):
    # Combinar todos los DataFrames y exportar  
    combined_df = pd.concat(lista_df, ignore_index=True)
    reorder_columns_df = combined_df[['layer','wordID','meaningID','target','sesgoBase','sesgoGen','error','meanError','standardError']] 
    reorder_rows_df = reorder_columns_df.sort_values(by=['wordID', 'meaningID', 'layer'], ascending=[True, True, True])
    reorder_rows_df.to_csv(f'{basepath}errorByLayer.csv', index=False)
    return reorder_rows_df

def get_errores_para_todas_las_capas(lista_de_df, layers, basepath):
    ## Para guardarme los errores en un csv
    df_con_errores = []
    for layer in layers:
        df = calculo_error(lista_de_df[layer])
        ## Para guardarme los errores en un csv
        df['layer'] = layer
        df_con_errores.append(df)
    df_combinado = getCombinedDFAndExport(df_con_errores, basepath)
    return df_combinado

def calculo_error(df_por_contexto):
    ## Calculo el promedio de las distancias ortogonales a la identidad contando a cada target con un contexto por separado
    ## Entonces debo sumar las distancias de un mismo target pero distintos contextos
    ## La cantidad de distancias va a ser el doble de la cantidad de targets porque para cada uno hay dos contextos
    distancias_por_contexto = df_por_contexto.apply(lambda x : calculo_distancia_entre_sesgoBase_sesgoGenerado(x), axis = 1)
    df_por_contexto['error'] = distancias_por_contexto
    df_por_contexto['meanError'] = distancias_por_contexto.mean()
    df_por_contexto['standardError'] = distancias_por_contexto.sem()
    return  df_por_contexto

def calculo_distancia_entre_sesgoBase_sesgoGenerado(row):
    ## Obtengo la distancia ortogonal a la identidad
    #TODO: sumar el valor absoluto del minimo
    e = 1
    return (row.sesgoGen - row.sesgoBase)

def getLayersMeanAndStandardError(df):
    generalError = df[(df['wordID'] == (df['wordID'].iloc[0])) & (df['meaningID'] == (df['meaningID'].iloc[0]))]
    layers = generalError['layer']
    meanErrors = generalError['meanError']
    standardErrors = generalError['standardError']
    return layers, meanErrors, standardErrors


def get_plot(df, basepath, titulo_segun):
    layers, meanErrors, standardErrors = getLayersMeanAndStandardError(df)
    plt.figure(figsize=(10,15))
    plt.errorbar(layers, meanErrors, yerr=(standardErrors))
    plt.title(f"Sesgos {titulo_segun}en cada capa de GPT2")
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

def get_plot_with_scatterplot(df, basepath, titulo_segun, size = (10,40), target_a_graficar = 'todos los targets', name = 'allTargets', word_id = ''):
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
    plt.title(f"Scatter plot de sesgos para {target_a_graficar} {titulo_segun}en cada capa de GPT2")
    plt.xlabel("Capa del modelo")
    plt.ylabel("Diferencia entre sesgo generado y sesgo base")
    plt.xticks(layers)
    plt.ylim(limiteMenor, limiteMayor)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{basepath}pltLayers_{name}_scatter.png')
    plt.close()

def get_plots_for_each_target(df, basepath, titulo_segun):
    unique_combinations = df[['wordID']].drop_duplicates()
    for word_id in unique_combinations.values:
        word_id = word_id[0]
        subset_word = df[(df['wordID'] == word_id)]
        target = subset_word['target'].unique()
        get_plot_with_scatterplot(df, basepath, titulo_segun, (10,15), f'el target {target}', f'target_{int(word_id)+1}', word_id)

def getPlots(lista_de_df, layers, basepath, titulo_segun):
    df = get_errores_para_todas_las_capas(lista_de_df, layers, basepath)
    ## Para graficar una linea con errores estandar
    get_plot(df, f'{basepath}plots/', titulo_segun)
    ## Para graficar con scatterplot
    get_plot_with_scatterplot(df, f'{basepath}plots/', titulo_segun)
    get_plots_for_each_target(df, f'{basepath}plots/', titulo_segun)
