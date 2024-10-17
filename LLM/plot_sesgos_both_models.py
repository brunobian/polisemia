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

# Load data for both models
df = pd.read_csv(f"versionGPT2(sinValorAbsoluto)/{meaningType}/errorByLayer.csv")
df_w = pd.read_csv(f"versionGPT2_wordlevel(sinValorAbsoluto)/{meaningType}/errorByLayer.csv")

# Add model identifier
df['Model'] = 'GPT2'
df_w['Model'] = 'GPT2 WordLevel'

# Combine both dataframes
df = pd.concat([df_w, df], ignore_index=True)

# Rename columns for clarity
df = df.rename(columns={
    'sesgoBase': 'baseS',
    'sesgoGen': 'SignS'
})

# Set global font sizes
plt.rcParams['font.size'] = 25  # Base font size
plt.rcParams['axes.titlesize'] = 30  # Title font size
plt.rcParams['axes.labelsize'] = 22  # Axis label font size

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
        global_min = min(x_min, y_min)
        global_max = max(x_max, y_max)
    elif plot_type == 'boxplot':
        y_min = -2 #data['diff_base_emb'].min()
        y_max = 2.5 #data['diff_base_emb'].max()
    
    for layer in data['layer'].unique():
        layer_data = data[data['layer'] == layer]
        
        plt.figure(figsize=(10, 15))
        
        if plot_type == 'scatter':
            for model in layer_data['Model'].unique():
                model_data = layer_data[layer_data['Model'] == model]
                plt.scatter(model_data['baseS'], model_data['SignS'], label=model)
            plt.plot([global_min, global_max], [global_min, global_max], 'k--')
            plt.xlim(global_min, global_max)
            plt.ylim(global_min, global_max)
            plt.xlabel("Base Bias")
            plt.ylabel("Generated Bias")
        
        elif plot_type == 'boxplot':
            sns.boxplot(x='Model', y='diff_base_emb', data=layer_data)
            plt.ylim(y_min, y_max)
            plt.ylabel("Difference between base and generated bias")
        
        plt.title(f"Layer {layer}")
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{plot_type}/layer_{layer}.png", dpi=300)
        plt.savefig(f"{output_dir}/{plot_type}/layer_{layer}.svg", dpi=300)
        plt.close()

# Create scatter plots for each layer
create_plots_by_layer(df, 'scatter')

# Calculate distance of biases to the diagonal
df['diff_base_emb'] = (df['SignS'] - df['baseS']) / df['baseS'].abs()

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
    y_min = -4 #data['diff_base_emb'].min()
    y_max = 7.5 #data['diff_base_emb'].max()
    # Use the most extreme value for both axes
    global_min = min(x_min, y_min)
    global_max = max(x_max, y_max)
    
    for layer in sorted(data['layer'].unique()):
        layer_data = data[data['layer'] == layer]
        
        plt.figure(figsize=(9, 12))
        for model in layer_data['Model'].unique():
            model_data = layer_data[layer_data['Model'] == model]
            plt.scatter(model_data['diff_base_human'], 
                       model_data['diff_base_emb'], 
                       label=model)

        plt.plot([global_min, global_max], [global_min, global_max], 'k--')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.axvline(x=0, color='k', linestyle=':')
        plt.axhline(y=0, color='k', linestyle=':')
        plt.xlabel("Human Bias (%)")
        plt.ylabel("Model Bias (%)")
        plt.title(f"Human vs Embedding Bias - Layer {layer}")
        plt.legend()
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/human_embedding/layer_{layer}.png", dpi=300)
        plt.savefig(f"{output_dir}/human_embedding/layer_{layer}.svg", dpi=300)
        plt.close()

# Create human vs embedding bias plots for each layer
create_human_embedding_plots(df)

# Function to perform statistical tests for each layer
def perform_statistical_tests(data):
    # Filter out layer 0
    data = data[data['layer'] != 0]
    results = []
    for layer in data['layer'].unique():
        layer_data = data[data['layer'] == layer]
        
        diff_base_emb_WL = layer_data[layer_data['Model'] == 'GPT2 WordLevel']['diff_base_emb']
        diff_base_emb_viejo = layer_data[layer_data['Model'] == 'GPT2']['diff_base_emb']
        
        # T-test to compare means of the two models
        test_modelos = stats.ttest_ind(diff_base_emb_WL, diff_base_emb_viejo, alternative='greater')
        
        # Use diff_base_human directly
        diff_base_humanos = layer_data['diff_base_human']
        
        # T-tests to compare model biases with human biases
        test_WL_Humano = stats.ttest_ind(diff_base_emb_WL, diff_base_humanos)
        test_Base_Humano = stats.ttest_ind(diff_base_emb_viejo, diff_base_humanos)
        
        results.append({
            'Layer': layer,
            'WordLevel vs base': test_modelos,
            'WordLevel vs Human': test_WL_Humano,
            'base vs Human': test_Base_Humano
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
        base_kde = stats.gaussian_kde(layer_data[layer_data['Model'] == 'GPT2']['diff_base_emb'].dropna())
        wl_kde = stats.gaussian_kde(layer_data[layer_data['Model'] == 'GPT2 WordLevel']['diff_base_emb'].dropna())
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
        plt.figure(figsize=(9, 15))
        densities = densities_by_layer[layer]
        plt.plot(densities['x_eval'], densities['base_density'], 
                color="red", linewidth=2, label="base")
        plt.plot(densities['x_eval'], densities['wl_density'], 
                color="blue", linewidth=2, label="Word Level")
        
        plt.fill_between(densities['x_eval'], densities['base_density'], 
                        alpha=0.2, color="red")
        plt.fill_between(densities['x_eval'], densities['wl_density'], 
                        alpha=0.2, color="blue")
        
        plt.xlim(-7, 7)
        plt.ylim(0, y_max)
        plt.xlabel("Difference between base and meaning embeddings")
        plt.ylabel("Density")
        plt.title(f"Density Plots - Layer {layer}")
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{output_dir}/density_plots/layer_{layer}.png", dpi=300)
        plt.savefig(f"{output_dir}/density_plots/layer_{layer}.svg", dpi=300)
        plt.close()

# Create density plots for each layer
create_density_plots(df)