import pandas as pd
from modifyStimuli import getStimuliAsDF 
from modifyExperimentResults import importAndCleanExperimentResults, importExperimentResults

def mergeStimuliAndResults(stimuli, results):
    merged_df = pd.merge(stimuli, results, on=['wordID','target','meaningID'], how='left').reset_index()
    merged_df['answers'] = merged_df.apply(lambda row: row['answers'] if isinstance(row['answers'], list) else row['significado'], axis=1)
    merged_df = merged_df.drop(columns=['index', 'significado'])
    merged_df = merged_df.rename(columns={'answers': 'significado'})
    return merged_df

def getStimuliMergedWithExperimentResults(stimuli, results):
    df = getStimuliAsDF(stimuli)
    resultsDf = importAndCleanExperimentResults(results)
    merged_df = mergeStimuliAndResults(df, resultsDf)
    return merged_df

def getStimuliMergedWithFormattedExperimentResults(stimuli, results):
    df = getStimuliAsDF(stimuli)
    resultsDf = importExperimentResults(results)
    merged_df = mergeStimuliAndResults(df, resultsDf)
    return merged_df
'''
DF RETURNED:
index  wordID      target  meaningID  contextID combinationID         significado                                       oracion                                  Contexto
    0       0        raya          0          3          3000  [("a",1), ("b",2)]  Daniel observó la raya fijamente durante ...  La tilde diacrítica es la que permite ...
    1       0        raya          0          1          1000           [("a",1)]  Daniel observó la raya fijamente durante ...  El mar, vasto e inexplorado, es hogar ...

Aclaraciones:
- El wordID corresponde al id del target/palabra ambigua
- El target es la palabra ambigua en minuscula
- El meaningID es el numero (comenzando en 0) del significado de la palabra ambigua. 
    Puede ser 0 o 1 y se corresponde con significado (una lista de palabras relacionadas)
- El contextID es el numero de contexto del stimuli. 
    El 1 corresponde al meaningID 0, el 2 con el meaningID 1 y el 3 seria un contexto que no tiene nada que ver (que sirve para analizar el sesgo base)
- CombinationID es CMWW con C el numero de contextID, M el meaningID y WW dos digitos para el wordID.
    Sirve para comparar con solo una columna y saber con que contexto y significado estoy trabajando
- significado es una lista de tuplas donde en la primera parte esta una palabra relacionada y en la segunda, la cantidad de veces que fue elegida.
    Sirve para poder tener un listado de palabras de significado y tomar el promedio ponderado de sus sesgos
- Oracion es la oracion ambigua
- Contexto son las oraciones previas a la oracion con la palabra ambigua. 
    Sirve para orientar el significado de la palabra: si se relaciona con el significado, se tiene el sesgo generado; y si no esta relacionado, es el sesgo base
'''