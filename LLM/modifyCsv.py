import pandas as pd
import json
from collections import Counter

def cargar_stimuli(stimuli_path):
    return pd.read_csv(stimuli_path,quotechar='"')

def dropAndRenameColumns(df):
    df = df.drop(columns=['Funciona','targetMAYUS','palabras mal usadas','palabras mal usadas.1','palabras mal usadas.2','Significado3','Unnamed: 14','Prompts'])
    df = df.rename(columns={'indTarget': 'wordID'})
    return df

def getDf_byMeaningAndContext(df):
    df_by_signif = pd.wide_to_long(df, stubnames='significado', i='wordID', j='meaningID').reset_index()
    df_by_context = pd.wide_to_long(df_by_signif, stubnames='Contexto', i=['wordID', 'meaningID'], j='contextID').reset_index()
    df_reordered = df_by_context[['wordID','target','oracion','meaningID','significado','contextID', 'Contexto']] 
    df_reordered = df_reordered.sort_values(by=['wordID', 'meaningID', 'contextID'], ascending=[True, True, False])
    return df_reordered

def renameContextID_byMeaningIDWordID(df, byMeaningID=False, byWordID=False):
    df['combinationID'] = (df['contextID']*1000+byMeaningID*df['meaningID']*100+byWordID*df['wordID']).astype(str)
    return df

def filterCombinations(df):
    ##Quiero mantener: 31-, 11-, 32-, 22-
    mask = df['combinationID'].str.startswith(('12','21'))
    return df[~mask]

def getStimuliAsDF(file):
    df = cargar_stimuli(file) 
    df = dropAndRenameColumns(df)
    df = getDf_byMeaningAndContext(df)
    df = renameContextID_byMeaningIDWordID(df, True, True)
    df = filterCombinations(df)
    return df

def loadExperimentResults(results):
    with open(results) as f:
        data = json.load(f)
    trials_list = []
    for entry in data:
        user_id = entry['userId']
        trials = entry['trials']
        for trial in trials:
            # Crea un diccionario con los datos del trial
            trial_data = {
                'userId': user_id,
                'trialNumber': trial['trialNumber'],
                'wordID': trial['wordID'],
                'meaningID': trial['meaningID'],
                'word': trial['word'],
                'context': trial['context'],
                'answers': trial['answers'],
                'wordOrder': trial['wordOrder'],
                'submitTime': trial['submitTime']['$date']
            }
            trials_list.append(trial_data)
    return pd.DataFrame(trials_list)

def getListOfAnswersToEachCombination(df):
    def combine_answers(series):
        combined_list = []
        for answer_list in series:
            combined_list.extend(answer_list)
        return combined_list  
    result = df.groupby(['wordID', 'meaningID', 'context'])['answers'].apply(combine_answers).reset_index()
    return result

def getDictionaryOfAnswers(df):
    dictionaries = []
    for listOfAnswers in df['answers']:
        # Contar las apariciones de cada string
        counter = Counter(listOfAnswers)
        # Convertir el contador en una lista de pares (string, cantidad)
        pairs = list(counter.items())
        # Ordenar la lista de pares por cantidad de apariciones (de mayor a menor)
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        dictionaries.append(sorted_pairs)
    df['answers'] = dictionaries        
    return df

def cleanResults(resultsDf):
    df = resultsDf.drop(columns= ['userId', 'trialNumber', 'wordOrder', 'submitTime'])
    df = df.sort_values(by=['wordID','meaningID','context'], ascending=[True, True, True])
    df = getListOfAnswersToEachCombination(df)
    df = getDictionaryOfAnswers(df)
    return df

def idsAreCorrelated(stimuliDf, resultDf):
    #TODO: Falta validar ids de wordId, meaningID, contextID
    return True

def mergeStimuliAndResults(stimuli, results):
    #TODO: completar
    return results

def getStimuliMergeWithExperimentResults(stimuli, results):
    df = getStimuliAsDF(stimuli)
    resultsDf = loadExperimentResults(results)
    resultsDf = cleanResults(resultsDf)
    assert(idsAreCorrelated(df, resultsDf))
    merged_df = mergeStimuliAndResults(df, resultsDf)
    return merged_df


df = getStimuliAsDF('Stimuli_conListaDeSignificadosConPeso.csv')
resultsDf = loadExperimentResults('test.experimentmodels.json')
resultsDf = cleanResults(resultsDf)
# Muestra el DataFrame resultante
print(resultsDf)

'''
DF RETURNED:
rid   wordID      target  meaningID    significado  contextID                                           Contexto
  0        0        raya          1     Animales;1        100  El mar, vasto e inexplorado, es hogar de una a...
'''

'''
Consultas:
- El context va a ser indicado con el texto o se puede agregar el contextID?
- Tenemos que chequear que wordID corresponda con el del stimuli, no?
- Los contextos son los que estaban?
- A que nos referimos cuando hablamos de meaning id? tiene sentido mantener los meanings del stimuli? 
- ver que queda y que se va
'''