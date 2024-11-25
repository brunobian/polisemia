import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import seaborn as sns
import matplotlib.pyplot as plt

def get_mixed_linear_model_results(data):
    modelo = mixedlm("error ~ layer", data, groups=data["wordID"])
    resultado = modelo.fit()
    #print(resultado.summary())
    return resultado

def get_plot_residuos(resultado, basepath):
    residuos = resultado.resid
    sns.histplot(residuos, kde=True)
    plt.title('Distribución de los residuos')
    plt.xlabel('Residuos')
    plt.savefig(f'modeloLinealMixto/{basepath}plot-residuos.png')
    plt.savefig(f'modeloLinealMixto/{basepath}plot-residuos.svg')
    plt.close()

def get_plot_efectos_fijos(resultado, basepath):
    effects_fixed = resultado.fe_params.reset_index()
    effects_fixed.columns = ['Layer', 'Effect']
    sns.barplot(x='Layer', y='Effect', data=effects_fixed)
    plt.title("Efectos Fijos por Capa")
    plt.xlabel("Capa")
    plt.ylabel("Efecto")
    plt.xticks(rotation=45)
    plt.savefig(f'modeloLinealMixto/{basepath}plot-efectos_fijos.png')
    plt.savefig(f'modeloLinealMixto/{basepath}plot-efectos_fijos.svg')
    plt.close()

def get_plot_efectos_random(resultado, basepath):
    random_effects = resultado.random_effects
    random_effects_df = pd.DataFrame(random_effects).reset_index()
    print(random_effects_df)
    sns.scatterplot(x='wordID', y='random_effects', data=random_effects_df)
    plt.title("Efectos Aleatorios por Palabra")
    plt.xlabel("ID de Palabra")
    plt.ylabel("Efecto Aleatorio")
    plt.xticks(rotation=45)
    plt.savefig(f'modeloLinealMixto/{basepath}plot-efectos_random.png')
    plt.savefig(f'modeloLinealMixto/{basepath}plot-efectos_random.svg')
    plt.close()

def get_plot_residuos_vs_valores_ajustados(resultado, basepath):
    residuals = resultado.resid
    fitted_values = resultado.fittedvalues
    plt.scatter(fitted_values, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuos vs. Valores Ajustados")
    plt.xlabel("Valores Ajustados")
    plt.ylabel("Residuos")
    plt.savefig(f'modeloLinealMixto/{basepath}plot-residuos_vs_valores_ajustados.png')
    plt.savefig(f'modeloLinealMixto/{basepath}plot-residuos_vs_valores_ajustados.svg')
    plt.close()

def get_plot_errores_por_capa(resultado, basepath):
    sns.boxplot(x='layer', y='error', data=resultado)
    plt.title("Distribución del Error por Capa")
    plt.xlabel("Capa")
    plt.ylabel("Error")
    plt.xticks(rotation=45)
    plt.savefig(f'modeloLinealMixto/{basepath}plot-errores_por_capa.png')
    plt.savefig(f'modeloLinealMixto/{basepath}plot-errores_por_capa.svg')
    plt.close()

def get_plot_intervalos_de_confianza(resultado, basepath):
    effects_fixed = resultado.fe_params.reset_index()
    effects_fixed.columns = ['Layer', 'Effect']
    # Obtener intervalos de confianza
    ci = resultado.conf_int()
    ci.columns = ['2.5%', '97.5%']
    # Asegúrate de que el índice de 'effects_fixed' sea compatible con 'ci'
    effects_fixed['Layer'] = effects_fixed['Layer'].astype(str)  # Convertir a string si es necesario
    ci = ci.reset_index()
    ci.columns = ['Layer', '2.5%', '97.5%']
    # Combinar efectos fijos con intervalos de confianza
    combined = pd.merge(effects_fixed, ci, on='Layer')
    plt.plot(combined['Layer'], combined['Effect'], marker='o', label='Efecto Fijo')
    plt.fill_between(combined['Layer'], combined['2.5%'], combined['97.5%'], alpha=0.2, label='Intervalo de Confianza')
    plt.title("Efectos Fijos con Intervalos de Confianza")
    plt.xlabel("Capa")
    plt.ylabel("Efecto")
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(f'modeloLinealMixto/{basepath}plot-intervalos_confianza.png')
    plt.savefig(f'modeloLinealMixto/{basepath}plot-intervalos_confianza.svg')
    plt.close()

def get_plot_mixed_linear_model(csv, basepath):
    data = pd.read_csv(csv)
    resultado = get_mixed_linear_model_results(data)
    get_plot_residuos(resultado, basepath)
    get_plot_efectos_fijos(resultado, basepath)
    #get_plot_efectos_random(resultado, basepath)
    get_plot_residuos_vs_valores_ajustados(resultado, basepath)
    get_plot_errores_por_capa(data, basepath)
    get_plot_intervalos_de_confianza(resultado, basepath)   

def plot_mixed_linear_model_by_meaning(meaning):
    basepath = f"{sinValorAbs}/{meaning}/"
    csv = f"versionGPT2({sinValorAbs})/{meaning}/errorByLayer.csv"
    get_plot_mixed_linear_model(csv, f'GPT2/{basepath}')
    csv = f"versionGPT2_wordlevel({sinValorAbs})/{meaning}/errorByLayer.csv"
    get_plot_mixed_linear_model(csv, f'GPT2_wordlevel/{basepath}')

    basepath = f"{valorAbs}/{meaning}/"
    csv = f"versionGPT2({valorAbs})/{meaning}/errorByLayer.csv"
    get_plot_mixed_linear_model(csv, f'GPT2/{basepath}')
    csv = f"versionGPT2_wordlevel({valorAbs})/{meaning}/errorByLayer.csv"
    get_plot_mixed_linear_model(csv, f'GPT2_wordlevel/{basepath}')

    basepath = f"{valorAbsSesgo}/{meaning}/"
    csv = f"versionGPT2({valorAbsSesgo})/{meaning}/errorByLayer.csv"
    get_plot_mixed_linear_model(csv, f'GPT2/{basepath}')
    csv = f"versionGPT2_wordlevel({valorAbsSesgo})/{meaning}/errorByLayer.csv"
    get_plot_mixed_linear_model(csv, f'GPT2_wordlevel/{basepath}')

def plot_mixed_linear_model():
    plot_mixed_linear_model_by_meaning(unMeaning)
    plot_mixed_linear_model_by_meaning(shuffleado)
    plot_mixed_linear_model_by_meaning(listaMeaning)

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

plot_mixed_linear_model()