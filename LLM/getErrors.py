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