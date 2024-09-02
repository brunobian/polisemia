import matplotlib.pyplot as plt

## Para las metricas

def get_errores_para_todas_las_capas(lista_de_df, layers):
    ## Para graficar una linea con errores estandar
    error_promedio_por_capa = []
    error_estandar_por_capa = []
    ## Para graficar con boxplot
    errores_por_capa = []
    for layer in layers:
        error_promedio, error_estandar, errores = calculo_error(lista_de_df[layer])
        ## Para graficar una linea con errores estandar
        error_promedio_por_capa.append(error_promedio)
        error_estandar_por_capa.append(error_estandar)
        ## Para graficar con boxplot tengo lista de listas
        errores_por_capa.append(errores)
    return errores_por_capa, error_promedio_por_capa, error_estandar_por_capa

def calculo_error(df_por_contexto):
    ## Calculo el promedio de las distancias ortogonales a la identidad contando a cada target con un contexto por separado
    ## Entonces debo sumar las distancias de un mismo target pero distintos contextos
    ## La cantidad de distancias va a ser el doble de la cantidad de targets porque para cada uno hay dos contextos
    distancias_por_contexto = df_por_contexto.apply(lambda x : calculo_distancia_entre_sesgoBase_sesgoGenerado(x), axis = 1)
    error_promedio = distancias_por_contexto.mean()
    error_estandar = distancias_por_contexto.sem()
    return error_promedio, error_estandar, distancias_por_contexto.to_list()

def calculo_distancia_entre_sesgoBase_sesgoGenerado(row):
    ## Obtengo la distancia ortogonal a la identidad
    #TODO: sumar el valor absoluto del minimo
    e = 1
    return (row.sesgoGen - row.sesgoBase)

def get_plot(distances, errores_estandar, layers):
    '''figsize=(10,10))'''
    plt.errorbar(layers, distances, yerr=(errores_estandar))
    plt.title("Sesgos segun cada capa de GPT2")
    plt.xlabel("Capas")
    plt.xticks(layers)
    plt.ylabel("Error promedio")
    plt.yticks(distances)
    plt.savefig('nuevo/pltLayers_-layers.png')
    plt.show()

def reordeno_por_targetcontexto (errores_por_capa):
    # Traspongo la matriz para que este ordenada por filas en vez de por capas
    # Errores por capa: capa x -> sesgo_gen - sesgo_base de cada fila
    # Quiero modificarlo para tener una lista de listas con:
    # - Cada lista se corresponde con una fila n 
    # - Para cada fila, esta sesgo_gen - sesgo_base de cada capa
    # - Queda [n, 13] porque son 12 filas mas la 0 (previa a aplicar el modelo)
    return [[fila[i] for fila in errores_por_capa ] for i in range(len(errores_por_capa[0]))] 

def get_plot_with_scatterplot(distances, errores_promedio, errores_estandar, layers, df, size = (10,40), target_a_graficar = 'todos los targets', name = 'allTargets', nombre_target = ''):
    colors = plt.cm.tab20.colors
    num_colors = len(distances)
    colors_extended = colors * (num_colors // len(colors)) + colors[:num_colors % len(colors)]
    plt.figure(figsize=size)
    for i in range(0, len(distances), 2):
        # Si tiene un target ingresado como parametro, uso ese (caso de grafico para solo un target)
        # Sino, tomo el del df (caso de todos los targets juntos)
        if(nombre_target == ''):
           target = df['target'][i]
        else:
            target = nombre_target
        # Genero el scatter plot para cada meaning del mismo target
        plt.scatter(layers, distances[i], color=colors_extended[i], label=f'Target {target} con meaning ID 0')  # Scatter plot para cada lista
        plt.scatter(layers, distances[i+1], color=colors_extended[i+1], label=f'Target {target} con meaning ID 1')  # Scatter plot para cada lista
    #for i_dist, lista_dist in enumerate(distances):
    #    plt.scatter(layers, lista_dist, c=colors_extended[i_dist], label=f'Target con meaning ID {i_dist}')  # Scatter plot para cada lista
    plt.errorbar(layers, errores_promedio, yerr=errores_estandar, fmt='o', color='black', capsize=5, label='Error estándar')
    plt.title(f"Scatter plot de sesgos para {target_a_graficar} según cada capa de GPT2")
    plt.xlabel("Capa del modelo")
    plt.ylabel("Diferencia entre sesgo generado y sesgo base")
    plt.xticks(layers)
    plt.legend()  # Mostrar leyenda con etiquetas de lista
    # Mostrar el gráfico
    plt.grid(True)
    plt.savefig(f'nuevo/pltLayers_{name}_scatter.png')
    plt.close()

def get_plots_for_each_target(distances, errores_promedio, errores_estandar, layers, df):
    for i in range(0, len(distances), 2):
        distances_for_each_target = [distances[i], distances[i+1]]
        target = df['target'][i]
        indiceTarget = (i/2)
        get_plot_with_scatterplot(distances_for_each_target, errores_promedio, errores_estandar, layers, df, (10,15), f'el target {target}', f'target_{indiceTarget+1}', target)

def get_plot_with_boxplot(distances, errores_promedio, errores_estandar, layers):
    plt.figure(figsize=(10,15))
    plt.boxplot(distances, positions=layers)
    plt.errorbar(layers, errores_promedio, yerr=(errores_estandar))
    plt.title("Sesgos segun cada capa de GPT2")
    plt.xlabel("Capas")
    plt.xticks(layers)
    plt.ylabel("Error promedio")
    #plt.yticks(errores_promedio)
    plt.savefig('nuevo/pltLayers_allLayers_boxplot.png')
    plt.show()

def getPlots(lista_de_df, layers):
    errores_por_capa, error_promedio_por_capa, error_estandar_por_capa = get_errores_para_todas_las_capas(lista_de_df, layers)
    ## Para graficar una linea con errores estandar
    #get_plot(error_promedio_por_capa, error_estandar_por_capa, layers)
    ## Para graficar con boxplot
    errores_por_target = reordeno_por_targetcontexto(errores_por_capa)
    get_plot_with_scatterplot(errores_por_target, error_promedio_por_capa, error_estandar_por_capa, layers, lista_de_df[0])
    get_plots_for_each_target(errores_por_target, error_promedio_por_capa, error_estandar_por_capa, layers, lista_de_df[0])
