from plots import get_plot_with_more_than_one_csv,get_plot_with_different_tokenizer, get_plot_with_different_cluster

sinValorAbs = "sinValorAbsoluto"
valorAbs = "valorAbsoluto"
valorAbsSesgo = "absDeCadaSesgo"

labelsinValorAbs = "Diferencia entre sesgos"
labelvalorAbs = "Absolute value of generated bias - basal bias"
labelvalorAbsSesgo = "Absolute value of generated bias - absolute value of basal bias"

soloUnaPalabra = "soloUnaPalabra"
shuffleado = "sinExperimento/conMeaningsAleatorios"
unMeaning = "sinExperimento/conMeaningsDeStimuli"
listaMeaning = "conExperimento(4taVersion)"

def plot_comparing_gpt2_and_gpt2wordlevel_one_type_meaning(titulo_segun, meaning):
    labels = ["GPT-2","GPT-2 Word Level"]

    listaCsv = [f"versionGPT2({sinValorAbs})/{meaning}/errorByLayer.csv", f"versionGPT2_wordlevel({sinValorAbs})/{meaning}/errorByLayer.csv"]
    basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{sinValorAbs}/{meaning}/"
    get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labels, labelsinValorAbs)

    listaCsv = [f"versionGPT2({valorAbs})/{meaning}/errorByLayer.csv", f"versionGPT2_wordlevel({valorAbs})/{meaning}/errorByLayer.csv"]
    basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{valorAbs}/{meaning}/"
    get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labels,labelvalorAbs)

    listaCsv = [f"versionGPT2({valorAbsSesgo})/{meaning}/errorByLayer.csv", f"versionGPT2_wordlevel({valorAbsSesgo})/{meaning}/errorByLayer.csv"]
    basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{valorAbsSesgo}/{meaning}/"
    get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labels, labelvalorAbsSesgo)

def plot_comparing_each_type_meaning():
    titulo_segun = "con respecto a solo una palabra "
    meaning = soloUnaPalabra
    plot_comparing_gpt2_and_gpt2wordlevel_one_type_meaning(titulo_segun, meaning)

    titulo_segun = "usando una categoria como significado "
    meaning = unMeaning
    plot_comparing_gpt2_and_gpt2wordlevel_one_type_meaning(titulo_segun, meaning)

    titulo_segun = "usando significados mezclados "
    meaning = shuffleado
    plot_comparing_gpt2_and_gpt2wordlevel_one_type_meaning(titulo_segun, meaning)

    titulo_segun = "usando lista de palabras como significado "
    meaning = listaMeaning
    plot_comparing_gpt2_and_gpt2wordlevel_one_type_meaning(titulo_segun, meaning)

def plot_one_word_vs_list_words():
    labelUnaOLista = ["GPT-2 usando una categoria","GPT-2 Word Level usando una categoria","GPT-2 usando lista de palabras","GPT-2 Word Level usando lista de palabras"]

    titulo_segun = "usando una categoria o lista de palabras "
    meaning1 = listaMeaning
    meaning2 = unMeaning
    meaning = "unoVsLista"
    listaCsv = [f"versionGPT2({sinValorAbs})/{meaning2}/errorByLayer.csv", f"versionGPT2_wordlevel({sinValorAbs})/{meaning2}/errorByLayer.csv", f"versionGPT2({sinValorAbs})/{meaning1}/errorByLayer.csv", f"versionGPT2_wordlevel({sinValorAbs})/{meaning1}/errorByLayer.csv"]
    basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{sinValorAbs}/{meaning}/"
    get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labelUnaOLista,labelsinValorAbs)

    listaCsv = [f"versionGPT2({valorAbs})/{meaning2}/errorByLayer.csv", f"versionGPT2_wordlevel({valorAbs})/{meaning2}/errorByLayer.csv", f"versionGPT2({valorAbs})/{meaning1}/errorByLayer.csv", f"versionGPT2_wordlevel({valorAbs})/{meaning1}/errorByLayer.csv"]
    basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{valorAbs}/{meaning}/"
    get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labelUnaOLista,labelvalorAbs)

    listaCsv = [f"versionGPT2({valorAbsSesgo})/{meaning2}/errorByLayer.csv", f"versionGPT2_wordlevel({valorAbsSesgo})/{meaning2}/errorByLayer.csv", f"versionGPT2({valorAbsSesgo})/{meaning1}/errorByLayer.csv", f"versionGPT2_wordlevel({valorAbsSesgo})/{meaning1}/errorByLayer.csv"]
    basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{valorAbsSesgo}/{meaning}/"
    get_plot_with_more_than_one_csv(listaCsv, basepath, titulo_segun, labelUnaOLista,labelvalorAbsSesgo)

def plot_different_tokenizers():
    labelTokenizadores = ["Usando lista de significados y promedio de tokens","Usando lista de significados y el primer token","Usando una categoría como significado y promedio de tokens","Usando una categoría como significado y el primer token"]

    titulo_segun = ""
    meaning1 = listaMeaning
    meaning2 = unMeaning
    listaCsv = [f"versionGPT2({sinValorAbs})/{meaning1}/tokenizadorPromedio/errorByLayer.csv", f"versionGPT2({sinValorAbs})/{meaning1}/tokenizadorPrimerToken/errorByLayer.csv", f"versionGPT2({sinValorAbs})/{meaning2}/tokenizadorPromedio/errorByLayer.csv", f"versionGPT2({sinValorAbs})/{meaning2}/tokenizadorPrimerToken/errorByLayer.csv"]
    basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{sinValorAbs}/tokenizadores/"
    get_plot_with_different_tokenizer(listaCsv, basepath, titulo_segun, labelTokenizadores,labelsinValorAbs)

def plot_different_clusters():
    labelClusters = ["Baja entropia","Alta entropia","Sin agrupar"]

    listaCsv = [f"versionGPT2_wordlevel({sinValorAbs})/clusters/primerCluster/errorByLayer.csv", f"versionGPT2_wordlevel({sinValorAbs})/clusters/segundoCluster/errorByLayer.csv",f"versionGPT2_wordlevel({sinValorAbs})/{listaMeaning}/errorByLayer.csv"]
    basepath = f"comparativaSesgos/gpt2yGpt2Wordlevel/{sinValorAbs}/clusters/"
    get_plot_with_different_cluster(listaCsv, basepath, "", labelClusters,labelsinValorAbs, "GPT-2_Word_Level")

plot_comparing_each_type_meaning()
plot_one_word_vs_list_words()
plot_different_tokenizers()
plot_different_clusters()
titulo_segun = ""
meaning = listaMeaning
#plot_comparing_gpt2_and_gpt2wordlevel_one_type_meaning(titulo_segun, meaning)

import pandas as pd

# Cargar el archivo CSV en un DataFrame
df1 = pd.read_csv(f"versionGPT2({sinValorAbs})/{listaMeaning}/errorByLayer.csv")
df2 = pd.read_csv(f"versionGPT2_wordlevel({sinValorAbs})/{listaMeaning}/errorByLayer.csv")

# Filtrar las filas donde 'sesgoBase' > 'sesgoGen'
resultado1 = df1[df1['sesgoGen'] > df1['sesgoBase']][['layer','target','meaningID','sesgoBase', 'sesgoGen']]
resultado1['diferenciaSesgos'] = df1['sesgoGen'] - df1['sesgoBase']
resultado2 = df2[df2['sesgoGen'] > df2['sesgoBase']][['layer','target','meaningID','sesgoBase', 'sesgoGen']]
resultado2['diferenciaSesgos'] = df2['sesgoGen'] - df2['sesgoBase']

# Filtrar las capas menores a 5
resultado1 = resultado1[resultado1['layer'] == 5]
#resultado1 = resultado1[resultado1['sesgoGen'] < -0.09]
resultado1 = resultado1.nlargest(5, 'sesgoBase')
resultado1_top10 = resultado1.sort_values(by=['layer', 'sesgoBase'])

resultado2 = resultado2[resultado2['layer'] == 5]
#resultado2 = resultado2[resultado2['sesgoGen'] < -0.09]
resultado2 = resultado2.nlargest(5, 'sesgoBase')
resultado2_top10 = resultado2.sort_values(by=['layer', 'sesgoBase'])

# Imprimir los resultados
print(f"Para el resultado de gpt 2 se observaron diferencias en: {resultado1_top10}")
print(f"Para el resultado de gpt 2 WL se observaron diferencias en: {resultado2_top10}")