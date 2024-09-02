import pandas as pd
import json
from collections import Counter

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
                'word': trial['word'].lower(),
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
    result = df.groupby(['wordID', 'meaningID', 'target'])['answers'].apply(combine_answers).reset_index()
    return result

def getDictionaryOfAnswers(df):
    dictionaries = []
    for listOfAnswers in df['answers']:
        # Paso los significados a minuscula
        listOfAnswers = [significado.lower() for significado in listOfAnswers]
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
    df = resultsDf.drop(columns= ['userId', 'trialNumber', 'context', 'wordOrder', 'submitTime'])
    df = df.sort_values(by=['wordID','meaningID'], ascending=[True, True])
    df = df.rename(columns={'word': 'target'})
    df = getListOfAnswersToEachCombination(df)
    df = getDictionaryOfAnswers(df)
    '''
    DF RETURNED:
    wordID    target  meaningID        answers  
         0      raya          1     Animales;1   
    '''
    return df

def importAndCleanExperimentResults(file):
    resultsDf = loadExperimentResults(file).reset_index()
    return cleanResults(resultsDf)