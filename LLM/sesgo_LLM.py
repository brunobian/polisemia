from transformers import LlamaTokenizerFast,AutoModelForCausalLM,GPT2Tokenizer  
from torch.nn import functional as f 
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
'''import pingouin as pg 
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison'''

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from tokenizers.processors import TemplateProcessing

'''
Observar que hay partes del codigo que estan comentadas, leer lo siguiente antes de ejecutar:
- En el metodo "cargar_modelo" revisar que el path a usar este en la maquina donde estes ejecutando
    - En la de Bruno: "/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish"
    - En la de Belu: "/Users/NaranjaX/Desktop/tesis/clm-spanish"
- El modelo para Llama2 puede instalarse cada vez o buscarse en un repo en caso de estar instalado
- Hay que usar ".to(cuda)" si usas la compu de Bruno y ".to("mps")" sino en la mac
'''

## Para preparar el experimento
def cargar_modelo(model_type, model_path=""):
    #TODO: Revisar que seria el modal path y cuando pasarlo
    #TODO: Sacar estos if usando algun patron de diseño
    if model_type == "GPT2":
        ruta      = "/Users/NaranjaX/Desktop/tesis/clm-spanish"#"/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish"
        model     = AutoModelForCausalLM.from_pretrained(ruta,device_map="auto")                  
        tokenizer = GPT2Tokenizer.from_pretrained(ruta)
    elif model_type == "Llama2":
        ruta = "/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/Llama-2-7b-hf/snapshots/3f025b66e4b78e01b4923d510818c8fe735f6f54"
        model = AutoModelForCausalLM.from_pretrained(ruta, device_map="auto",load_in_8bit=True)    
        #model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", device_map="auto",load_in_8bit=True,cache_dir="/data/brunobian/languageModels/")    
        tokenizer = LlamaTokenizerFast.from_pretrained(ruta)                            
    elif model_type == "GPT2_wordlevel":
        tokenizer_dict_path = '/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/models/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish_word_stimulidb+reddit/tokenizer/token_dict.pkl'
        ruta = '/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/models/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish_word_stimulidb+reddit/checkpoint-29160'
        model = AutoModelForCausalLM.from_pretrained(ruta).to("mps")#.to('cuda')
        try:
            with open(tokenizer_dict_path, 'rb') as tokenizer_file:
                tokenizer_dict = pickle.load(tokenizer_file)
                word2int = tokenizer_dict['word2int']
                int2word = tokenizer_dict['int2word']
                unk_token = tokenizer_dict['UNK']
        except Exception as e:
            print(f"Failed to load tokenizer dictionary: {e}")

        tokenizer = Tokenizer(WordLevel(vocab = word2int, unk_token = unk_token))
        tokenizer.pre_tokenizer = Sequence([Punctuation('removed'), Whitespace()])
    else:
        print("modelo incorrecto")
    #TODO: Ver que nos gustaria devolver
    return model, tokenizer
    
def cargar_stimuli(stimuli_path):
    return pd.read_csv(stimuli_path,quotechar='"')

def get_significados_sin_peso(row):
    significados = []
    signif1 = []
    for lista_signif1 in row.significado1.split(','):
        signif1.append(lista_signif1.split(';')[0].lower())
    significados.append(signif1)
    signif2 = []
    for lista_signif2 in row.significado2.split(','):
        signif2.append(lista_signif2.split(';')[0].lower())
    significados.append(signif2)
    return significados

def get_pesos(row):
    pesos = []
    pesos1 = []
    for lista_signif1 in row.significado1.split(','):
        pesos1.append(int(lista_signif1.split(';')[1]))
    pesos.append(pesos1)
    pesos2 = []
    for lista_signif2 in row.significado2.split(','):
        pesos2.append(int(lista_signif2.split(';')[1]))
    pesos.append(pesos2)
    return pesos

def get_iterador (row):
    sig = get_significados_sin_peso(row)
    pesos = get_pesos(row)
    context = [row.Contexto3, row.Contexto1, row.Contexto2]
    iterar_sobre = [(sig[0], pesos[0], context[0]), (sig[0], pesos[0], context[1]), (sig[1], pesos[1], context[0]), (sig[1], pesos[1], context[2])]
    return iterar_sobre

## Para la capa 0
def get_embedding_before_model(model, model_type, target):
    if model_type == "GPT2":
        target_embedding = model.transformer.wte(target[0])
    elif model_type == "Llama2":
        target_embedding = model.get_input_embeddings().weight[target]
    elif model_type == "GPT2_wordlevel":
        target_embedding = model.transformer.wte(target[0])
    
    stacked_target_embeddings = torch.stack([target_embedding])
    target_embeddings = torch.nanmean(stacked_target_embeddings, dim=0)

    return target_embeddings

## Para el resto de las capas
def get_query(context, oracion):
    query = context + " " + oracion + "."
    return query

def tokenize(text, tokenizer, model_type):
    if model_type == "GPT2_wordlevel":
        text_ids = torch.tensor([tokenizer.encode(text).ids]).to("mps")#.to('cuda')
    else:
        text_ids = tokenizer.encode(text, return_tensors="pt").to("mps")#.to('cuda') 
    return text_ids

def find_target(text_ids, target, lista_sig, model_type, tokenizer):
    ## Obtengo el target tokenizado y en formato tensor
    if model_type == "GPT2":
        targ_tokenizado = tokenizer.encode(" " + target) # GPT
        target_token  = tokenizer.encode(" " + target, return_tensors="pt").to("mps")#.to('cuda') 
    elif model_type== "Llama2":
        targ_tokenizado = tokenizer.encode(target)[1:] # Llama
        target_token  = tokenizer.encode(target, return_tensors="pt").to("mps")#.to('cuda')
    elif model_type == "GPT2_wordlevel":
        targ_tokenizado = tokenizer.encode(target) # GPT
        target_token  = torch.tensor([tokenizer.encode(target).ids]).to("mps")#.to('cuda')
            
    ## Obtengo los significados de la lista tokenizados en formato tensor
    list_sig_ids = []
    for sig in lista_sig:
        if model_type == "GPT2":
            # TODO: leer como funciona el encode si sig fuese una lista
            sig_ids  = tokenizer.encode(" " + sig, return_tensors="pt").to("mps")#.to('cuda') 
            sig_ids_list= sig_ids[0].tolist()  # si la palabra del significado tiene mas de un token, me quedo con el primero
        elif model_type== "Llama2":
            sig_ids  = tokenizer.encode(sig, return_tensors="pt").to("mps")#.to('cuda')
            sig_ids  = sig_ids[0,1:]  # Saco el primer elemento que es <s>
            sig_ids_list= sig_ids.tolist() 
        elif model_type == "GPT2_wordlevel":
            sig_ids  = torch.tensor([tokenizer.encode(sig).ids]).to("mps")#.to('cuda')
            sig_ids_list= sig_ids.tolist()  # si la palabra del significado tiene mas de un token, me quedo con el primero
        list_sig_ids.append(sig_ids)        
        
    ## Busco las posiciones del target en el texto
    text_ids_list = text_ids[0].tolist()
    for i in range(len(text_ids_list)):
        if targ_tokenizado == text_ids_list[i:(i+len(targ_tokenizado))]:
            break
    ids_target_in_text = list(range(i,i+len(targ_tokenizado)))
    #print(tokenizer.decode(text_ids[ids_target_in_text]))
    return list_sig_ids, ids_target_in_text, target_token

def find_significado(text_ids, text_ids_list, sig_ids_list):
    #Busco las posiciones del significado en el texto
    for i in range(len(text_ids_list)):                                 
        if sig_ids_list == text_ids_list[i:(i+len(sig_ids_list))]:              
            break                                                       
    ids_sig_in_text = list(range(i,i+len(sig_ids_list))) 
    print(tokenizer.decode(text_ids[ids_sig_in_text]))  

def instance_model(text_ids, model):
    # Corro el modelo para el texto 
    # y extraigo embedding del target contextualizado
    output_text = model(text_ids, output_hidden_states=True)
    h = output_text.hidden_states 
    return h

def extract_embedding_from_layer(hidden_state, ids_target_in_text, layer=-1):
    last_layer  = hidden_state[layer][0]
    target_embeddings = last_layer[ids_target_in_text]
    #sig_embeddings    = last_layer[ids_sig_in_text]
    return target_embeddings

def get_sig_embedding(model, model_type, sig, pesoDeSignificados):
    ## Version apta para listas de significados (donde tomo el promedio de los embeddings)
    sig_embeddings_list = []

    if model_type == "GPT2":
        for s in sig:
            sig_embeddings_list.append(model.transformer.wte(s[0])) #Tomo el primer id para cada palabra de la lista de significado
    elif model_type == "Llama2":
        for s in sig:
            sig_embeddings_list.append(model.get_input_embeddings().weight[s]) #Aca no tome el primero porque no estaba en el codigo original
    elif model_type == "GPT2_wordlevel":
        for s in sig:
            sig_embeddings_list.append(model.transformer.wte(s[0])) #Tomo el primer id para cada palabra de la lista de significado

    '''?????????? ACA VA CON EL PROMEDIO PONDERADO ??????????'''
    stacked_embeddings = torch.stack(sig_embeddings_list)
    pesoDeSignificados = torch.tensor(pesoDeSignificados).to("mps")#.to('cuda')
    stacked_embeddings_pesados = stacked_embeddings * pesoDeSignificados.unsqueeze(1)
    sig_embeddings = torch.sum(stacked_embeddings_pesados, dim=0)
    #stacked_embeddings = torch.stack(sig_embeddings_list)
    #sig_embeddings = torch.nanmean(stacked_embeddings, dim=0)

    # Corro el modelo para el significado
    # output_sig = model(sig_ids, output_hidden_states=True)            
    # last_layer  = output_sig.hidden_states[-2:][0].nanmean(0) 
    # sig_embeddings = last_layer[1:]

    return sig_embeddings

def get_diference_target_sig(target_embeddings, sig_embeddings):
    target_av_embedding = target_embeddings.nanmean(0)
    '''?????????? ACA VA CON EL PROMEDIO PONDERADO ??????????'''
    sig_av_embedding    = sig_embeddings.nanmean(0)
    dist = f.cosine_similarity(target_av_embedding,sig_av_embedding,0).item()

    #dist = max([f.cosine_similarity(target_embeddings[i,:],sig_embeddings[j,:],0).item()
    #     for i in range(target_embeddings.size(0)) 
    #            for j in range(sig_embeddings.size(0))])
    return dist

## Para obtener los sesgos con el modelo una vez por fila
def get_diference_multiple_context_all_layer(list_sig_y_contexto, target, oracion, layers):
    sesgos_en_capa_dada = []
    for listaSignificados, pesoDeSignificados, contexto in list_sig_y_contexto:
        sesgos_por_fila = []
        #pregunta = "En la oración anterior el significado de la palabra " + target + " está asoacido a " + s + "." # TODO: Ver cuando se usaria esto
        query = get_query(contexto, oracion)
        text_ids = tokenize(query, tokenizer, m)
        sig_ids, ids_target_in_text, target_token = find_target(text_ids, target, listaSignificados, m, tokenizer) # TODO: No esta en uso find_significado, si lo saco, eliminar lo que devuelvo y no se usa
        #find_significado(text_ids, text_ids[0].tolist(), sig_ids.tolist()) # No esta en uso find_significado, estaba comentado, ver si lo dejo o lo saco
        hidden_state = instance_model(text_ids, model) 
        sig_embeddings = get_sig_embedding(model, m, sig_ids, pesoDeSignificados)
        for layer in layers:
            if layer == 0:
                target_embeddings = get_embedding_before_model(model, m, target_token)
                sesgos_por_fila.append(get_diference_target_sig(target_embeddings, sig_embeddings))
            else:
                target_embeddings = extract_embedding_from_layer(hidden_state, ids_target_in_text, layer)
                sesgos_por_fila.append(get_diference_target_sig(target_embeddings, sig_embeddings))
        sesgos_en_capa_dada.append(sesgos_por_fila)
        ## En este punto tengo, por cada target-contexto: [sesgo_layer_0, sesgo_layer_1, ..., sesgo_layer_12]
    return sesgos_en_capa_dada ## En este punto tengo, [sesgos_target-context_1, sesgos_target-context_2, sesgos_target-context_3, sesgos_target-context_4]

def get_sesgo_de_todas_capas_por_fila(row, layers):
    return get_diference_multiple_context_all_layer(get_iterador(row), row.target, row.oracion, layers)

def reordeno_matriz_sesgos(sesgos):
    '''
    tenemos:
        [fila 1: [Target-Contexto 1: [sc0tc1f1, sc1tc1f1, ... , sc12tc1f1],      
                  Target-Contexto 2: [sc0tc2f1, sc1tc2f1, ... , sc12tc2f1],  
                  Target-Contexto 3: [sc0tc3f1, sc1tc3f1, ... , sc12tc3f1],  
                  Target-Contexto 4: [sc0tc4f1, sc1tc4f1, ... , sc12tc4f1]], 
        ...
        fila N:  [Target-Contexto 1: [sc0tc1fN, sc1tc1fN, ... , sc12tc1fN],      
                  Target-Contexto 2: [sc0tc2fN, sc1tc2fN, ... , sc12tc2fN],  
                  Target-Contexto 3: [sc0tc3fN, sc1tc3fN, ... , sc12tc3fN],  
                  Target-Contexto 4: [sc0tc4fN, sc1tc4fN, ... , sc12tc4fN]]]
    queremos:
        [sesgos_capa_0: [fila_1: [sc0tc1f1, sc0tc2f1, sc0tc3f1, sc0tc4f1], 
                         ...
                         fila_N: [sc0tc1fN, sc0tc2fN, sc0tc3fN, sc0tc4fN]],
         ... , 
         sesgos_capa_12:[fila_1: [sc12tc1f1, sc12tc2f1, sc12tc3f1, sc12tc4f1], 
                         ...
                         fila_N: [sc12tc1fN, sc12tc2fN, sc12tc3fN, sc12tc4fN]],]
    '''
    sesgos_de_tc_por_capa_por_fila = []
    for sesgos_de_capa_por_targetcontext in sesgos:   
        ## Traspongo la matriz de capas por target-context a matriz de target-context por capas
        sesgos_de_targetcontext_por_capa = [[fila[i] for fila in sesgos_de_capa_por_targetcontext ] for i in range(len(sesgos_de_capa_por_targetcontext[0]))]
        sesgos_de_tc_por_capa_por_fila.append(sesgos_de_targetcontext_por_capa)
    ## Traspongo la matriz de capas por filas a matriz de filas por capas
    sesgos_por_capa = [[fila[i] for fila in sesgos_de_tc_por_capa_por_fila ] for i in range(len(sesgos_de_tc_por_capa_por_fila[0]))]
    return sesgos_por_capa

def get_lista_de_df(sesgos_por_capa):
    lista_de_df = []
    for sesgo_por_fila in sesgos_por_capa:
        lista_de_df.append(get_df_de_sesgo_del_modelo(sesgo_por_fila))
    return lista_de_df

## Para las metricas
def calculo_distancia_entre_sesgoBase_sesgoGenerado(row):
    ## Obtengo la distancia ortogonal a la identidad
    #TODO: sumar el valor absoluto del minimo
    e = 1
    return (row.sesgoGen - row.sesgoBase)

def calculo_error(df_por_contexto):
    #df_por_contexto.to_csv(f"distancias_por_layer.csv")
    ## Calculo el promedio de las distancias ortogonales a la identidad contando a cada target con un contexto por separado
    ## Entonces debo sumar las distancias de un mismo target pero distintos contextos
    ## La cantidad de distancias va a ser el doble de la cantidad de targets porque para cada uno hay dos contextos
    distancias_por_contexto = df_por_contexto.apply(lambda x : calculo_distancia_entre_sesgoBase_sesgoGenerado(x), axis = 1)
    error_promedio = distancias_por_contexto.mean()
    error_estandar = distancias_por_contexto.sem()
    return error_promedio, error_estandar, distancias_por_contexto.to_list()

def get_df_de_sesgo_del_modelo(all_sesgos, name=''):
    df=[]
    ## Para tener un df con una fila por palabra de target (quedan dos contextos por fila)
    #titulos =["fila", "numContexto1", "sesgoBase1", "sesgoGen1", "numContexto2", "sesgoBase2", "sesgoGen2" ]
    ## Para tener un df con una fila por palabra de target (quedan dos contextos por fila)
    titulos =["fila","numContexto", "sesgoBase", "sesgoGen"]
    for i,p in enumerate(all_sesgos):
        ## Para tener un df con una fila por palabra de target (quedan dos contextos por fila)
        #df.append([i, 1, p[0], p[1], 2, p[2], p[3]]) 
        ## Para tener un df con una fila por palabra de target+contexto
        df.append([i, 1, p[0], p[1]])
        df.append([i, 2, p[2], p[3]])

    df=pd.DataFrame(df, columns=titulos)
    #df.to_csv(f"distancias_por_layer_{name}.csv")
    return df

def get_plot(distances, errores_estandar, layers):
    '''figsize=(10,10))'''
    plt.errorbar(layers, distances, yerr=(errores_estandar))
    plt.title("Sesgos segun cada capa de GPT2")
    plt.xlabel("Capas")
    plt.xticks(layers)
    plt.ylabel("Error promedio")
    plt.yticks(distances)
    plt.savefig('pltLayers_-layers.png')
    plt.show()

def get_plot_with_boxplot(distances, errores_promedio, errores_estandar, layers):
    plt.figure(figsize=(10,15))
    plt.boxplot(distances, positions=layers)
    plt.errorbar(layers, errores_promedio, yerr=(errores_estandar))
    plt.title("Sesgos segun cada capa de GPT2")
    plt.xlabel("Capas")
    plt.xticks(layers)
    plt.ylabel("Error promedio")
    #plt.yticks(errores_promedio)
    plt.savefig('pltLayers_allLayers_boxplot.png')
    plt.show()

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

def get_plot_with_scatterplot(distances, errores_promedio, errores_estandar, layers, size = (10,40), nameTarget = 'todos los targets', name = 'allTargets'):
    colors = plt.cm.tab20.colors
    num_colors = len(distances)
    colors_extended = colors * (num_colors // len(colors)) + colors[:num_colors % len(colors)]
    plt.figure(figsize=size)
    for i_dist, lista_dist in enumerate(distances):
        plt.scatter(layers, lista_dist, c=colors_extended[i_dist], label=f'Target-contexto {i_dist}')  # Scatter plot para cada lista
    plt.errorbar(layers, errores_promedio, yerr=errores_estandar, fmt='o', color='black', capsize=5, label='Error estándar')
    plt.title(f"Scatter plot de sesgos para {nameTarget} según cada capa de GPT2")
    plt.xlabel("Capas")
    plt.ylabel("Errores")
    plt.xticks(layers)
    plt.legend()  # Mostrar leyenda con etiquetas de lista
    # Mostrar el gráfico
    plt.grid(True)
    plt.savefig(f'graficos/pltLayers_{name}_scatter.png')
    plt.close()

def reordeno_por_targetcontexto (errores_por_capa):
    return [[fila[i] for fila in errores_por_capa ] for i in range(len(errores_por_capa[0]))] # traspongo la matriz para que este ordenada por filas en vez de por capas

def get_plots_for_each_target(distances, errores_promedio, errores_estandar, layers, df):
    for i in range(0, len(distances), 2):
        distances_for_each_target = [distances[i], distances[i+1]]
        indiceTarget = (i/2)
        target = df['target'][indiceTarget]
        get_plot_with_scatterplot(distances_for_each_target, errores_promedio, errores_estandar, layers, (10,15), f'el target {target}', f'target_{indiceTarget+1}')

'''def control_de_caso_base(lista_de_listas_de_distancias):
    ## Preparar tus datos: Carga tus datos en un DataFrame de Pandas. Supongamos que tienes un conjunto de datos con una columna de grupos y una columna de valores (por ejemplo, resultados de un experimento):
    grupo = []
    valores = []
    for i in range(12):
        grupo.append(i)
        grupo.append(i)
        valores.append(valor0)
        valores.append(valori+1)
    data = pd.DataFrame({ 'grupo': grupo, 'valor': valores })
    ## Primero, ajusta un modelo usando ols (ordinary least squares)
    modelo = ols('valor ~ grupo', data=data).fit()
    ## Luego, realiza el ANOVA usando anova_lm con tipo 2 porque ese es el tipo de la suma de cuadrados usada
    ## El resultado del ANOVA (anova_resultados) te dirá si hay diferencias significativas entre los grupos.
    anova_resultados = sm.stats.anova_lm(modelo, typ=2) 
    ## Para realizar contrastes post hoc como el método de Tukey para comparaciones múltiples entre grupos
    mc = MultiComparison(data['valor'], data['grupo']) 
    ## Los resultados del método de Tukey (resultado_tukey) te mostrarán qué grupos difieren significativamente entre sí después de realizar el ANOVA.
    resultado_tukey = mc.tukeyhsd() 
    print(resultado_tukey)'''

m = "GPT2"#"Llama2"#
model, tokenizer = cargar_modelo(m) #TODO: Pensar mejor como devolver esto, si hace falta estar pasando las tres cosas o que
df = cargar_stimuli("Stimuli_conListaDeSignificadosConPeso.csv") #"Stimuli.csv"
layers = [0,1,2,3,4,5,6,7,8,9,10,11,12]
## Para obtener los sesgos ejecutando el modelo la menor cantidad de veces
sesgos_de_capas_por_fila = (df.apply(lambda r: get_sesgo_de_todas_capas_por_fila(r, layers), axis=1))
sesgos_por_capa = reordeno_matriz_sesgos(sesgos_de_capas_por_fila)
lista_de_df = get_lista_de_df(sesgos_por_capa)
## Para graficar una linea con errores estandar
#errores_por_capa, error_promedio_por_capa, error_estandar_por_capa = get_errores_para_todas_las_capas(lista_de_df, layers)
#get_plot(error_promedio_por_capa, error_estandar_por_capa, layers)
## Para graficar con boxplot
errores_por_capa, error_promedio_por_capa, error_estandar_por_capa = get_errores_para_todas_las_capas(lista_de_df, layers)
errores_por_target = reordeno_por_targetcontexto(errores_por_capa)
get_plot_with_scatterplot(errores_por_target, error_promedio_por_capa, error_estandar_por_capa, layers)
get_plots_for_each_target(errores_por_target, error_promedio_por_capa, error_estandar_por_capa, layers, df)
## Controles de enova
#titulos =["distCapa0", "distCapa1", "distCapa2", "distCapa3", "distCapa4", "distCapa5", "distCapa6", "distCapa7", "distCapa8", "distCapa9", "distCapa10", "distCapa11", "distCapa12" ]
#df_errores_por_capa = pd.DataFrame(errores_por_capa, columns=titulos)
#errores_control_de_caso_base = control_de_caso_base(errores_por_capa)
#errores_control_de_dosis = control_de_dosis(lista_de_df)

'''
OK Cambiar el nombre del eje y "Error promedio"
OK Agregar la capa 0 la distancia entre embeddings estaticos y el significado
OKHacer esto mismo para la target: get_sig_embedding
OK hacer reshape del df con los resultados para tener todo en una misma columna
OK Ademas del promedio(df.mean(columna)) calcular el error estandar (df.sem(columna))
OK Usar:
    plt.errorbar(capas, promedio de capa, yerr=(error estandar), fmt="o")
    plt.show()

buscar una palabra que se sample mal, re bien y promedio
analizar misma palabra distintos contextos
OK agregar en la resta la division por sesgo base para normalizar
NOSumamos uno a cada sesgo porque la distancia coseno va de -1 a 1 y queremos evitar los valores negativos
Buscar sobre anova (analisis de varianza) entre la capa interna y la de todas las capas
t de student
    Bonferroni para corregir los significativos: podria ser muy conservador
    entonces conviene usar enova con contrastes:
        control de caso base: osea entre el 0 y cada capa
        control de dosis: entre capa consecutivas
        Poner los dos plots por separado y agregar la mediana al boxplot
pasar todo a dataframe
guardame las distancias y acomodar el modelo para usar eso

CONSULTAS
como aplico anova? para diferenciar como se comporta la diferencia entre la capa 0 y las otras para cada fila? 
o tomo el promerio/error de cada capa y ls comparo con la capa 0?
Que resultado quiero tener? 
requiere post hoc?
que tipo de contraste aplico?
'''