from transformers import LlamaTokenizerFast,AutoModelForCausalLM,GPT2Tokenizer
from tokenizers import Tokenizer
from cambio_tokenizador.custom_tokenizer import CustomTokenizer   
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from tokenizers.processors import TemplateProcessing

from tqdm import tqdm
import pickle
'''import pingouin as pg 
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison'''


from modifyInput import getStimuliMergedWithExperimentResults, getStimuliMergedWithFormattedExperimentResults, getStimuliMergedWithFormattedExperimentResultsFromCluster, shuffleMeanings, setMeaning
from plots import getPlots
from getBias import getBias
from getBias_forPreparedDf import getBias_forPreparedDf
'''
Observar que hay partes del codigo que estan comentadas, leer lo siguiente antes de ejecutar:
- En el metodo "cargar_modelo" revisar que el path a usar este en la maquina donde estes ejecutando
    - En la de Bruno: 
        ruta de gpt2 -> "/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish"
        ruta de gpt2-worldlevel -> '/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/model/checkpoint-21940'
        ruta de tokenizer de gpt2-worldlevel ->'/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/tokenizer'
    - En la de Belu: 
        ruta de gpt2 -> "/Users/NaranjaX/Desktop/tesis/clm-spanish"
        ruta de gpt2-worldlevel -> '/Users/NaranjaX/Desktop/tesis/modeloAgus/checkpoint-21940'
        ruta de tokenizer de gpt2-worldlevel -> '/Users/NaranjaX/Desktop/tesis/modeloAgus/tokenizer'
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
        ruta = '/Users/NaranjaX/Desktop/tesis/modeloAgus/checkpoint-21940' #'/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/model/checkpoint-21940'
        tokenizer_path = '/Users/NaranjaX/Desktop/tesis/modeloAgus/tokenizer' #'/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/tokenizer'
        tokenizer = CustomTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(ruta).to("mps")#.to('cuda')
        '''VERSION DE AGUS ARRIBA
        tokenizer_dict_path = '/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/models/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish_word_stimulidb+reddit/tokenizer/token_dict.pkl'
        ruta = '/Users/NaranjaX/Desktop/tesis/modeloAgus/checkpoint-21940'#'/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/models/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish_word_stimulidb+reddit/checkpoint-29160'
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
        tokenizer.pre_tokenizer = Sequence([Punctuation('removed'), Whitespace()])'''
    else:
        print("modelo incorrecto")
    #TODO: Ver que nos gustaria devolver
    return model, tokenizer

layers = [0,1,2,3,4,5,6,7,8,9,10,11,12]

### GPT 2 ###
#m = "GPT2"
m = "GPT2_wordlevel"
version = "ConExperimento4" 
#"ConExperimento2" 
#"ConExperimento3" 
#"ConExperimento4" 
#"Shuffleada"
#"Hola"
#"SoloStimuli"
#"SoloUnaPalabraEnElModelo"
#Cluster1
#Cluster2
tipoMetrica = "sinValorAbsoluto"
#"valorAbsoluto"
#"sinValorAbsoluto"
#"absDeCadaSesgo"

if(version == "ConExperimento2"):
    basepath = f'version{m}({tipoMetrica})/conExperimento(2daVersion)/'
    titulo_segun = "segun lista de palabras significado "
    df = getStimuliMergedWithFormattedExperimentResults('Stimuli.csv', f'{basepath}experiment_results_formatted.csv', basepath)
else: 
    if(version == "ConExperimento3"):
        basepath = f'version{m}({tipoMetrica})/conExperimento(3erVersion)/'
        titulo_segun = "segun lista de palabras significado "
        df = getStimuliMergedWithExperimentResults('Stimuli.csv', f'{basepath}test.experimentmodels-final.json', basepath)
    else: 
        if(version == "ConExperimento4"):
            basepath = f'version{m}({tipoMetrica})/conExperimento(4taVersion)/'
            titulo_segun = "segun lista de palabras significado "
            df = getStimuliMergedWithFormattedExperimentResults('Stimuli.csv', f'{basepath}10-5-experiment_results_formatted.csv', basepath)
        else: 
            if(version == "Shuffleada"):
                basepath = f'version{m}({tipoMetrica})/sinExperimento/conMeaningsAleatorios/'
                titulo_segun = "segun un significado shuffleado "
                df = getStimuliMergedWithExperimentResults('Stimuli.csv', 'testVacio.experiment.json', basepath)
                df = shuffleMeanings(df, basepath)
            else: 
                if(version == "Hola"):
                    basepath = f'version{m}({tipoMetrica})/sinExperimento/conMeaningHola/'
                    titulo_segun = "segun el significado 'hola' "
                    df = getStimuliMergedWithExperimentResults('Stimuli.csv', 'testVacio.experiment.json', basepath)
                    df = setMeaning(df, 'hola', basepath)
                else: 
                    if(version == "SoloStimuli"):
                        basepath = f'version{m}({tipoMetrica})/sinExperimento/conMeaningsDeStimuli/'
                        titulo_segun = "segun una sola palabra significado "
                        df = getStimuliMergedWithExperimentResults('Stimuli.csv', 'testVacio.experiment.json', basepath)
                    else: 
                        if(version == "SoloUnaPalabraEnElModelo"):
                            basepath = f'version{m}({tipoMetrica})/soloUnaPalabra/'
                            titulo_segun = "a partir de solo la palabra raya "
                            df = getStimuliMergedWithExperimentResults(f'{basepath}StimuliUnaPalabra.csv', 'testVacio.experiment.json', basepath)
                        else: 
                            if(version == "Cluster1"):
                                basepath = f'version{m}({tipoMetrica})/clusters/primerCluster/'
                                titulo_segun = "con los resultados del primer cluster de participantes "
                                df = getStimuliMergedWithFormattedExperimentResultsFromCluster('Stimuli.csv', f'version{m}({tipoMetrica})/clusters/experiment_results_formatted_k_means.csv', basepath, 0)
                            else: 
                                if(version == "Cluster2"):
                                    basepath = f'version{m}({tipoMetrica})/clusters/segundoCluster/'
                                    titulo_segun = "con los resultados del segundo cluster de participantes "
                                    df = getStimuliMergedWithFormattedExperimentResultsFromCluster('Stimuli.csv', f'version{m}({tipoMetrica})/clusters/experiment_results_formatted_k_means.csv', basepath, 1)

model, tokenizer = cargar_modelo(m) #TODO: Pensar mejor como devolver esto, si hace falta estar pasando las tres cosas o que

#VERSION VIEJA (una fila por target)
#df = cargar_stimuli("Stimuli_conListaDeSignificadosConPeso.csv") #"Stimuli.csv"
#lista_de_df = getBias(df, layers, m, model, tokenizer)

#VERSION CON EXPERIMENTOS (una fila por combinacion target-significado-contexto)
lista_de_df = getBias_forPreparedDf(df, layers, m, model, tokenizer, basepath)
getPlots(lista_de_df, layers, basepath, titulo_segun, m, tipoMetrica)

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
OKpasar todo a dataframe

-Pendientes-
OKPasar a que el csv sea que en cada fila este target+contexto
usar el id num_comtexto*100+id_target para identificarlo (no el propio de pandas)
hacer merge con el json de tota -> Agregar la columna answers
pasar el pandas a csv
OKse usa pandas.wide_to_long para pasar de target+contexto1+contexto2 a target+contexto, target+contexto
Tenerlo como id_palabra(el del csv) | contexto | numeroDeContexto | idtrial(que se forme con numeroDeContexto*100+id_palabra)

Ver lo de modelos lineales mixtos
Pasar el cvs a id_trial | sesgo| capa y ejecutar en R con sesgo ~

COSAS COPADAS A ANALISAR:
- Revisar capa por capa como van variando sinonimos en un mismo contexto. Comparar la funcion que traza la evolucion del embedding de cada uno
'''