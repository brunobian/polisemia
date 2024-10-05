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

def get_errores_para_todas_las_capas(lista_de_df, layers, basepath, tipoMetrica):
    ## Para guardarme los errores en un csv
    df_con_errores = []
    for layer in layers:
        df = calculo_error(lista_de_df[layer], tipoMetrica)
        ## Para guardarme los errores en un csv
        df['layer'] = layer
        df_con_errores.append(df)
    df_combinado = getCombinedDFAndExport(df_con_errores, basepath)
    return df_combinado

def calculo_error(df_por_contexto, tipoMetrica):
    ## Calculo el promedio de las distancias ortogonales a la identidad contando a cada target con un contexto por separado
    ## Entonces debo sumar las distancias de un mismo target pero distintos contextos
    ## La cantidad de distancias va a ser el doble de la cantidad de targets porque para cada uno hay dos contextos
    distancias_por_contexto = df_por_contexto.apply(lambda x : calculo_distancia_entre_sesgoBase_sesgoGenerado(x, tipoMetrica), axis = 1)
    df_por_contexto['error'] = distancias_por_contexto
    df_por_contexto['meanError'] = distancias_por_contexto.mean()
    df_por_contexto['standardError'] = distancias_por_contexto.sem()
    return  df_por_contexto

def calculo_distancia_entre_sesgoBase_sesgoGenerado(row, tipoMetrica):
    ## Obtengo la distancia ortogonal a la identidad
    #TODO: sumar el valor absoluto del minimo
    e = 1
    sesgoGen = row.sesgoGen
    sesgoBase = row.sesgoBase
    if(tipoMetrica == "absDeCadaSesgo"):
        sesgoGen = abs(sesgoGen)
        sesgoBase = abs(sesgoBase)
    difSesgos = sesgoGen - sesgoBase
    if(tipoMetrica == "valorAbsoluto"):
        difSesgos = abs(difSesgos)
    return difSesgos

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

def getPlots(lista_de_df, layers, basepath, titulo_segun, m, tipoMetrica):
    df = get_errores_para_todas_las_capas(lista_de_df, layers, basepath, tipoMetrica)
    ## Para graficar una linea con errores estandar
    get_plot(df, f'{basepath}plots/', titulo_segun, m)
    ## Para graficar con scatterplot
    get_plot_with_scatterplot(df, f'{basepath}plots/', titulo_segun, m)
    get_plots_for_each_target(df, f'{basepath}plots/', titulo_segun, m)

def openCsvAsDf(listCsv):
    listDf = []
    for csv in listCsv:
        listDf.append(pd.read_csv(csv,quotechar='"'))
    return listDf

def get_plot_with_more_than_one_csv(listCsv, basepath, titulo_segun, labels):
    listDf = openCsvAsDf(listCsv)    
    colors_extended = getColors(len(listCsv))
    plt.figure(figsize=(10,15))
    for i,df in enumerate(listDf):
        layers, meanErrors, standardErrors = getLayersMeanAndStandardError(df)
        plt.errorbar(layers, meanErrors, yerr=(standardErrors), 
                    label=labels[i],
                    color=colors_extended[i])
    plt.title(f"Comparativa de sesgos {titulo_segun}en cada capa de los modelos")
    plt.xlabel("Capas")
    plt.ylabel("Diferencia entre sesgo generado y sesgo base")
    plt.legend(loc='best')
    #plt.ylim(-0.02, 0.035)
    plt.savefig(f'{basepath}plot-layers.png')
    plt.savefig(f'{basepath}plot-layers.svg')
    plt.close()
''''''
sinValorAbs = "sinValorAbsoluto"
valorAbs = "valorAbsoluto"
valorAbsSesgo = "absDeCadaSesgo"

soloUnaPalabra = "soloUnaPalabra"
shuffleado = "sinExperimento/conMeaningsAleatorios"
unMeaning = "sinExperimento/conMeaningsDeStimuli"
listaMeaning = "conExperimento(4taVersion)"

labelSinValorAbsoluto = ["GPT 2 sin valor absoluto","GPT 2 WordLevel sin valor absoluto"]
labelConValorAbsoluto = ["GPT 2 con valor absoluto a diferencia","GPT 2 WordLevel con valor absoluto a diferencia"]
labelConAbsolutoASesgos = ["GPT 2 con valor absoluto a sesgos","GPT 2 WordLevel con valor absoluto a sesgos"]

titulo_segun = "segun solo una palabra "
meaning = soloUnaPalabra
listaCsv = [f"versionGPT2({sinValorAbs})/{meaning}/errorByLayer.csv", f"versionGPT2_wordlevel({sinValorAbs})/{meaning}/errorByLayer.csv"]
basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{sinValorAbs}/{meaning}/"
get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labelSinValorAbsoluto)

listaCsv = [f"versionGPT2({valorAbs})/{meaning}/errorByLayer.csv", f"versionGPT2_wordlevel({valorAbs})/{meaning}/errorByLayer.csv"]
basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{valorAbs}/{meaning}/"
get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labelConValorAbsoluto)

listaCsv = [f"versionGPT2({valorAbsSesgo})/{meaning}/errorByLayer.csv", f"versionGPT2_wordlevel({valorAbsSesgo})/{meaning}/errorByLayer.csv"]
basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{valorAbsSesgo}/{meaning}/"
get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labelConAbsolutoASesgos)


titulo_segun = "segun solo una palabra como significado "
meaning = unMeaning
listaCsv = [f"versionGPT2({sinValorAbs})/{meaning}/errorByLayer.csv", f"versionGPT2_wordlevel({sinValorAbs})/{meaning}/errorByLayer.csv"]
basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{sinValorAbs}/{meaning}/"
get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labelSinValorAbsoluto)

listaCsv = [f"versionGPT2({valorAbs})/{meaning}/errorByLayer.csv", f"versionGPT2_wordlevel({valorAbs})/{meaning}/errorByLayer.csv"]
basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{valorAbs}/{meaning}/"
get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labelConValorAbsoluto)

listaCsv = [f"versionGPT2({valorAbsSesgo})/{meaning}/errorByLayer.csv", f"versionGPT2_wordlevel({valorAbsSesgo})/{meaning}/errorByLayer.csv"]
basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{valorAbsSesgo}/{meaning}/"
get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labelConAbsolutoASesgos)


titulo_segun = "segun significados aleatorios "
meaning = shuffleado
listaCsv = [f"versionGPT2({sinValorAbs})/{meaning}/errorByLayer.csv", f"versionGPT2_wordlevel({sinValorAbs})/{meaning}/errorByLayer.csv"]
basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{sinValorAbs}/{meaning}/"
get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labelSinValorAbsoluto)

listaCsv = [f"versionGPT2({valorAbs})/{meaning}/errorByLayer.csv", f"versionGPT2_wordlevel({valorAbs})/{meaning}/errorByLayer.csv"]
basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{valorAbs}/{meaning}/"
get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labelConValorAbsoluto)

listaCsv = [f"versionGPT2({valorAbsSesgo})/{meaning}/errorByLayer.csv", f"versionGPT2_wordlevel({valorAbsSesgo})/{meaning}/errorByLayer.csv"]
basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{valorAbsSesgo}/{meaning}/"
get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labelConAbsolutoASesgos)


titulo_segun = "segun significados aleatorios "
meaning = listaMeaning
listaCsv = [f"versionGPT2({sinValorAbs})/{meaning}/errorByLayer.csv", f"versionGPT2_wordlevel({sinValorAbs})/{meaning}/errorByLayer.csv"]
basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{sinValorAbs}/{meaning}/"
get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labelSinValorAbsoluto)

listaCsv = [f"versionGPT2({valorAbs})/{meaning}/errorByLayer.csv", f"versionGPT2_wordlevel({valorAbs})/{meaning}/errorByLayer.csv"]
basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{valorAbs}/{meaning}/"
get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labelConValorAbsoluto)

listaCsv = [f"versionGPT2({valorAbsSesgo})/{meaning}/errorByLayer.csv", f"versionGPT2_wordlevel({valorAbsSesgo})/{meaning}/errorByLayer.csv"]
basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{valorAbsSesgo}/{meaning}/"
get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labelConAbsolutoASesgos)
