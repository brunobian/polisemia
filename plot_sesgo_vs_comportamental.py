import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


lmmPath = "LLM/versionGPT2/"
human = pd.read_csv("comportamental/accuracies.csv")

human_1 = human[["indTarget", "diff_base1"]]
human_1.columns = ["wordID", "diffSesgo_humano"]
human_1["meaningID"] = 0
human_2 = human[["indTarget", "diff_base2"]]
human_2.columns = ["wordID", "diffSesgo_humano"]
human_2["meaningID"] = 1

human = pd.concat([human_1,human_2])
todo = human

corrs = []
for c in range(13):
    print(c)
    orig = pd.read_csv(f"{lmmPath}sinExperimento/conMeaningsDeStimuli/sesgos_por_layer_{c}.csv")
    orig["diffSesgo"] = abs(orig["sesgoGen"] - orig["sesgoBase"] )
    orig["Medición"] = "Original"
    orig.drop(["Unnamed: 0", "target", "sesgoBase", "sesgoGen"], axis=1, inplace=True)

    tmp = pd.merge(orig, human, on = ["wordID", "meaningID"])
    r_orig, p_orig = stats.pearsonr(tmp["diffSesgo_humano"], tmp["diffSesgo"])
    #p_orig = p_orig<.05
    
    nuevo = pd.read_csv(f"{lmmPath}conExperimento(3erVersion)/sesgos_por_layer_{c}.csv")
    nuevo["diffSesgo"] = abs(nuevo["sesgoGen"] - nuevo["sesgoBase"] )
    nuevo["Medición"] = "Nuevo"
    nuevo.drop(["Unnamed: 0", "target", "sesgoBase", "sesgoGen"], axis=1, inplace=True) 

    tmp = pd.merge(nuevo, human, on = ["wordID", "meaningID"])
    r_nuevo, p_nuevo = stats.pearsonr(tmp["diffSesgo_humano"], tmp["diffSesgo"])
    #p_nuevo = p_nuevo<.05

    corrs.append([c, r_orig, p_orig, r_nuevo, p_nuevo])

    df_capa = pd.concat([orig,nuevo])
    df_capa["capa"] = c

    if c==0:
        todo = df_capa
    else:
        todo = pd.concat([todo, df_capa])


import seaborn as sns
sns.lineplot(data=todo, x="capa", y="diffSesgo", hue="Medición")
plt.show()

plt.clf()
df = pd.DataFrame(corrs, columns = ['Capa', 'Corr Orig', 'P Orig', 'Corr Nuevo', 'P Nuevo']) 

plt.plot(df['Capa'], df['Corr Orig'], color='red', label='Original')
plt.plot(df['Capa'], df['Corr Nuevo'], color='blue', label='Nuevo')
# Adding labels and legend
plt.xlabel('Capa')
plt.ylabel('Correlacón con Sesgo Humano')
plt.legend()

plt.show()

pass
'''
# Plotting
plt.scatter(todo['diffSesgo_humano'], todo['diffSesgo_nuevo'], color='blue', label='Nuevo')
plt.scatter(todo['diffSesgo_humano'], todo['diffSesgo_orig'], color='green', label='Orig')

# Adding labels and legend
plt.xlabel('Diferencia de Sesgo (humano)')
plt.ylabel('Diferencia de Sesgo (GPT2)')
plt.legend()

# Display plot
plt.show()

plt.boxplot([todo['diffSesgo_nuevo'], todo['diffSesgo_orig']], labels=['Nuevo', 'Orig'])
# Adding labels
plt.ylabel('Sesgo Computacional')

# Display the plot
plt.show()

pass
'''