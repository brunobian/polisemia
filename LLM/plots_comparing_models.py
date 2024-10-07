from plots import get_plot_with_more_than_one_csv

sinValorAbs = "sinValorAbsoluto"
valorAbs = "valorAbsoluto"
valorAbsSesgo = "absDeCadaSesgo"

labelsinValorAbs = "Difference between generated bias and baseline bias"
labelvalorAbs = "Absolute value of generated bias - baseline bias"
labelvalorAbsSesgo = "Absolute value of generated bias - absolute value of baseline bias"

soloUnaPalabra = "soloUnaPalabra"
shuffleado = "sinExperimento/conMeaningsAleatorios"
unMeaning = "sinExperimento/conMeaningsDeStimuli"
listaMeaning = "conExperimento(4taVersion)"

def plot_comparing_gpt2_and_gpt2wordlevel_one_type_meaning(titulo_segun, meaning):
    labels = ["GPT 2","GPT 2 WordLevel"]

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
    titulo_segun = "according just one word "
    meaning = soloUnaPalabra
    plot_comparing_gpt2_and_gpt2wordlevel_one_type_meaning(titulo_segun, meaning)

    titulo_segun = "using one word as meaning "
    meaning = unMeaning
    plot_comparing_gpt2_and_gpt2wordlevel_one_type_meaning(titulo_segun, meaning)

    titulo_segun = "using shuffled meanings "
    meaning = shuffleado
    plot_comparing_gpt2_and_gpt2wordlevel_one_type_meaning(titulo_segun, meaning)

    titulo_segun = "using list of words as meanings "
    meaning = listaMeaning
    plot_comparing_gpt2_and_gpt2wordlevel_one_type_meaning(titulo_segun, meaning)

def plot_one_word_vs_list_words():
    labelUnaOLista = ["GPT 2 using one word","GPT 2 WordLevel using one word","GPT 2 con una lista de palabra","GPT 2 WordLevel con una lista de palabra"]

    titulo_segun = "using one word or a list of meanings "
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

plot_comparing_each_type_meaning()
plot_one_word_vs_list_words()