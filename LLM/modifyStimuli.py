import pandas as pd

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
    df['combinationID'] = df['contextID'].astype(str) + df['meaningID'].astype(str) + df['wordID'].astype(str).str.zfill(2)
    return df

def filterCombinations(df):
    ##Quiero mantener: 31-, 11-, 32-, 22-
    mask = df['combinationID'].str.startswith(('10','12','20','21','30'))
    return df[~mask]

def formatMeaning(row):
    signifList = []
    for lista_signif in row.split(','):
        signif = lista_signif.split(';')[0].lower()
        if(';' in lista_signif):
            peso = lista_signif.split(';')[1]
            assert(float(peso) - int(peso) == 0)
        else:
            peso = '1'  
        signifList.append((signif, int(peso)))
    return signifList

def replaceMeaningFormat(df):
    df['significado'] = df['significado'].apply(formatMeaning)
    return df

def getStimuliAsDF(file):
    df = cargar_stimuli(file) 
    df = dropAndRenameColumns(df)
    df = getDf_byMeaningAndContext(df)
    df = renameContextID_byMeaningIDWordID(df, True, True)
    df = filterCombinations(df)
    df = replaceMeaningFormat(df)
    '''
    DF RETURNED:
    rid   wordID      target  meaningID     significado  contextID                                           Contexto  combinationID
      0        0        raya          1  [(Animales,1)]          1  El mar, vasto e inexplorado, es hogar de una a...           1100
    '''
    return df