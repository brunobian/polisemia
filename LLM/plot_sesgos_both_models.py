import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Set working directory (uncomment and modify as needed)
# os.chdir("~/Desktop/Tesis/polisemia/LLM/")

# Set type of meaning
meaningType = "conExperimento(4taVersion)"
#"conExperimento(4taVersion)" 
#"sinExperimento/conMeaningsAleatorios"
#"sinExperimento/conMeaningHola"
#"sinExperimento/conMeaningsDeStimuli"
#"soloUnaPalabra"

# Create output directory
output_dir = f"comparativaSesgos/gpt2yGpt2Wordlevel/sinValorAbsoluto/{meaningType}"
'''
# Load data for both models
df = pd.read_csv(f"versionGPT2(sinValorAbsoluto)/{meaningType}/errorByLayer.csv")
df_w = pd.read_csv(f"versionGPT2_wordlevel(sinValorAbsoluto)/{meaningType}/errorByLayer.csv")

# Add model identifier
df['Model'] = 'GPT-2'
df_w['Model'] = 'GPT-2 Word Level'

# Combine both dataframes
df = pd.concat([df_w, df], ignore_index=True)
'''
df = pd.read_csv(f"versionGPT2_wordlevel(sinValorAbsoluto)/{meaningType}/errorByLayer.csv")
df['Model'] = 'GPT-2 Word Level'

# Rename columns for clarity
df = df.rename(columns={
    'sesgoBase': 'baseS',
    'sesgoGen': 'SignS'
})

# Set global font sizes
plt.rcParams['font.size'] = 12  # Base font size
plt.rcParams['axes.titlesize'] = 10  # Title font size
plt.rcParams['axes.labelsize'] = 15  # Axis label font size
LABELSIZE = 15 
TITLESIZE = 15
FIGSIZE_SCATTER_MODEL = (7, 6)
FIGSIZE_BOXPLOT_MODEL = (9, 14)
FIGSIZE_SCATTER_HUMAN = (7, 6)
FIGSIZE_DENSITY_HUMAN = (9, 14)

# Function to create and save plots for each layer
def create_plots_by_layer(data, plot_type):
    # Filter out layer 0
    data = data[data['layer'] != 0]
    if plot_type == 'scatter':
        x_min = data['baseS'].min()
        x_max = data['baseS'].max()
        y_min = data['SignS'].min()
        y_max = data['SignS'].max()
        # Use the most extreme value for both axes to make them equal
        global_min = min(x_min, y_min)-0.01
        global_max = max(x_max, y_max)+0.01
    elif plot_type == 'boxplot':
        y_min = -2 #data['diff_base_emb'].min()
        y_max = 2.5 #data['diff_base_emb'].max()
    
    for layer in data['layer'].unique():
        layer_data = data[data['layer'] == layer]
        
        
        if plot_type == 'scatter':
            plt.figure(figsize=FIGSIZE_SCATTER_MODEL)
            for model in layer_data['Model'].unique():
                COLOR = 'blue' if model == 'GPT-2 Word Level' else 'red'
                model_data = layer_data[layer_data['Model'] == model]
                plt.scatter(model_data['baseS'], model_data['SignS'], label=model, color = COLOR)
            plt.plot([global_min, global_max], [global_min, global_max], 'k--')
            plt.xlim(global_min, global_max)
            plt.ylim(global_min, global_max)
            #plt.xlabel("Sesgo base")
            #plt.ylabel("Sesgo generado")
            plt.legend(loc='lower right', fontsize=TITLESIZE)
        
        elif plot_type == 'boxplot':
            plt.figure(figsize=FIGSIZE_BOXPLOT_MODEL)
            sns.boxplot(x='Model', y='diff_base_emb', data=layer_data)
            plt.ylim(y_min, y_max)
            plt.ylabel("Diferencia entre el sesgo generado y sesgo base")
        
        plt.text(0.1, 0.95, f"Capa {layer}", fontsize=TITLESIZE, ha='center', va='center', transform=plt.gca().transAxes)
        # Add description in the bottom right
        plt.tick_params(axis='both', which='major', labelsize=LABELSIZE)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{plot_type}/layer_{layer}.png", dpi=300)
        plt.savefig(f"{output_dir}/{plot_type}/layer_{layer}.svg", dpi=300)
        plt.close()

# Create scatter plots for each layer
create_plots_by_layer(df, 'scatter')

# Calculate distance of biases to the diagonal
df['diff_base_emb'] = (df['SignS'] - df['baseS'])

# Create boxplots for each layer
create_plots_by_layer(df, 'boxplot')

# Read data with behavioral results
df_humanos = pd.read_csv("../comportamental/accuracies.csv", sep=",")
df_humanos = df_humanos[['indTarget', 'diff_base1', 'diff_base2']]
df_humanos[['diff_base1', 'diff_base2']] = df_humanos[['diff_base1', 'diff_base2']].astype(float)

# Crear dos dataframes separados para cada meaningID
df_humanos_1 = df_humanos[['indTarget', 'diff_base1']].rename(columns={'indTarget': 'wordID', 'diff_base1': 'diff_base_human'})
df_humanos_1['meaningID'] = 1

df_humanos_2 = df_humanos[['indTarget', 'diff_base2']].rename(columns={'indTarget': 'wordID', 'diff_base2': 'diff_base_human'})
df_humanos_2['meaningID'] = 2

# Combinar los dataframes
df_humanos_combined = pd.concat([df_humanos_1, df_humanos_2], ignore_index=True)

# Realizar el merge con el dataframe principal
df = pd.merge(df, df_humanos_combined, 
              on=['wordID', 'meaningID'], 
              how='left')

df.to_csv(f"{output_dir}/df_merged.csv")

# Function to create human vs embedding bias plots for each layer
def create_human_embedding_plots(data):
    # Filter out layer 0
    data = data[data['layer'] != 0]
    # Calculate global min and max values for consistent scaling
    x_min = data['diff_base_human'].min()-0.05
    x_max = data['diff_base_human'].max()+0.05
    y_min = -0.075 #data['diff_base_emb'].min()
    y_max = 0.17 #data['diff_base_emb'].max()
    # Use the most extreme value for both axes
    global_min = min(x_min, y_min)
    global_max = max(x_max, y_max)
    
    for layer in sorted(data['layer'].unique()):
        layer_data = data[data['layer'] == layer]
        
        plt.figure(figsize=FIGSIZE_SCATTER_HUMAN)
        for model in layer_data['Model'].unique():
            COLOR = 'blue' if model == 'GPT-2 Word Level' else 'red'
            model_data = layer_data[layer_data['Model'] == model]
             # Calculate Pearson correlation
            correlation, p_value = stats.pearsonr(
                model_data['diff_base_human'],
                model_data['diff_base_emb']
            )
            # Plot scatter points
            plt.scatter(
                model_data['diff_base_human'], 
                model_data['diff_base_emb'], 
                color=COLOR,
                label=f"r = {correlation:.2f}, p = {p_value:.2f}"
            )
        #plt.plot([global_min, global_max], [global_min, global_max], 'k--')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.axvline(x=0, color='k', linestyle=':')
        plt.axhline(y=0, color='k', linestyle=':')
        #plt.xlabel("Sesgo en humanos")
        #plt.ylabel("Sesgo computacional")
        #plt.title(f"Sesgo en humano vs sesgo computacional - Capa {layer}")
        plt.text(0.1, 0.95, f"Capa {layer}", fontsize=TITLESIZE, ha='center', va='center', transform=plt.gca().transAxes)
        plt.legend()
        plt.tick_params(axis='both', which='major', labelsize=LABELSIZE)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/human_embedding/layer_{layer}.png", dpi=300)
        plt.savefig(f"{output_dir}/human_embedding/layer_{layer}.svg", dpi=300)
        plt.close()

# Create human vs embedding bias plots for each layer
create_human_embedding_plots(df)

'''
# Function to perform statistical tests for each layer
def perform_statistical_tests(data):
    # Filter out layer 0
    data = data[data['layer'] != 0]
    results = []
    for layer in data['layer'].unique():
        layer_data = data[data['layer'] == layer]
        
        diff_base_emb_WL = layer_data[layer_data['Model'] == 'GPT-2 Word Level']['diff_base_emb']
        diff_base_emb_viejo = layer_data[layer_data['Model'] == 'GPT-2']['diff_base_emb']
        
        # T-test to compare means of the two models
        test_modelos = stats.ttest_ind(diff_base_emb_WL, diff_base_emb_viejo, alternative='greater')
        
        # Use diff_base_human directly
        diff_base_humanos = layer_data['diff_base_human']
        
        # T-tests to compare model biases with human biases
        test_WL_Humano = stats.ttest_ind(diff_base_emb_WL, diff_base_humanos)
        test_Base_Humano = stats.ttest_ind(diff_base_emb_viejo, diff_base_humanos)
        
        results.append({
            'Layer': layer,
            'WordLevel vs GPT2': test_modelos,
            'WordLevel vs Human': test_WL_Humano,
            'GPT2 vs Human': test_Base_Humano
        })
    
    return pd.DataFrame(results)
# Perform statistical tests for each layer
test_results = perform_statistical_tests(df)

# Save test results to CSV
test_results.to_csv(f"{output_dir}/statistical_test_results.csv")

# Function to create density plots for each layer
def create_density_plots(data):
    # Filter out layer 0
    data = data[data['layer'] != 0]
    # Calculate global min and max values for consistent scaling
    x_min = data['diff_base_emb'].min()
    x_max = data['diff_base_emb'].max()
    # Pre-calculate all densities to find global y max
    y_max = 0
    densities_by_layer = {}
    for layer in sorted(data['layer'].unique()):
        layer_data = data[data['layer'] == layer]
        # Calculate kernel density estimates for both models
        base_kde = stats.gaussian_kde(layer_data[layer_data['Model'] == 'GPT-2']['diff_base_emb'].dropna())
        wl_kde = stats.gaussian_kde(layer_data[layer_data['Model'] == 'GPT-2 Word Level']['diff_base_emb'].dropna())
        # Create evaluation points
        x_eval = np.linspace(x_min, x_max, 200)
        # Evaluate densities
        base_density = base_kde(x_eval)
        wl_density = wl_kde(x_eval)
        # Store densities for later plotting
        densities_by_layer[layer] = {
            'x_eval': x_eval,
            'base_density': base_density,
            'wl_density': wl_density
        }
        # Update global y_max
        y_max = max(y_max, base_density.max(), wl_density.max())
    for layer in data['layer'].unique():
        plt.figure(figsize=FIGSIZE_DENSITY_HUMAN)
        densities = densities_by_layer[layer]
        plt.plot(densities['x_eval'], densities['base_density'], 
                color="red", linewidth=2, label="GPT-2")
        plt.plot(densities['x_eval'], densities['wl_density'], 
                color="blue", linewidth=2, label="GPT-2 Word Level")
        
        plt.fill_between(densities['x_eval'], densities['base_density'], 
                        alpha=0.2, color="red")
        plt.fill_between(densities['x_eval'], densities['wl_density'], 
                        alpha=0.2, color="blue")
        
        plt.xlim(-0.075, 0.17)
        plt.ylim(0, y_max)
        plt.xlabel("Diferencia entre sesgos")
        plt.ylabel("Densidad")
        #plt.title(f"Grafico de densidad - Capa {layer}")
        plt.tick_params(axis='both', which='major', labelsize=LABELSIZE)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{output_dir}/density_plots/layer_{layer}.png", dpi=300)
        plt.savefig(f"{output_dir}/density_plots/layer_{layer}.svg", dpi=300)
        plt.close()

# Create density plots for each layer
create_density_plots(df)'''