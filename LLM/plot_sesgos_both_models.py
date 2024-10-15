import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
df['Model'] = 'GPT2 base'
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
    for layer in data['layer'].unique():
        layer_data = data[data['layer'] == layer]
        
        plt.figure(figsize=(20, 30))
        
        if plot_type == 'scatter':
            for model in layer_data['Model'].unique():
                model_data = layer_data[layer_data['Model'] == model]
                plt.scatter(model_data['baseS'], model_data['SignS'], label=model)
            
            plt.plot([layer_data['baseS'].min(), layer_data['baseS'].max()], 
                     [layer_data['baseS'].min(), layer_data['baseS'].max()], 'k--')
            plt.xlabel("Base Bias")
            plt.ylabel("Generated Bias")
        
        elif plot_type == 'boxplot':
            sns.boxplot(x='Model', y='diff_base_emb', data=layer_data)
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

# Merge dataframes
df = pd.merge(df, df_humanos, left_on="wordID", right_on="indTarget")

# Function to create human vs embedding bias plots for each layer
def create_human_embedding_plots(data):
    for layer in data['layer'].unique():
        layer_data = data[data['layer'] == layer]
        
        plt.figure(figsize=(24, 18))
        for model in layer_data['Model'].unique():
            model_data = layer_data[layer_data['Model'] == model]
            plt.scatter(model_data[model_data['meaningID'] == 1]['diff_base1'], 
                        model_data[model_data['meaningID'] == 1]['diff_base_emb'], 
                        label=f'{model} S1')
            plt.scatter(model_data[model_data['meaningID'] == 2]['diff_base2'], 
                        model_data[model_data['meaningID'] == 2]['diff_base_emb'], 
                        label=f'{model} S2')

        plt.plot([layer_data[['diff_base1', 'diff_base2']].min().min(), 
                  layer_data[['diff_base1', 'diff_base2']].max().max()], 
                 [layer_data[['diff_base1', 'diff_base2']].min().min(), 
                  layer_data[['diff_base1', 'diff_base2']].max().max()], 'k--')
        plt.axvline(x=0, color='k', linestyle=':')
        plt.axhline(y=0, color='k', linestyle=':')
        plt.xlabel("Human Bias (%)")
        plt.ylabel("Embedding Bias (%)")
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
    results = []
    for layer in data['layer'].unique():
        layer_data = data[data['layer'] == layer]
        
        diff_base_emb_WL = layer_data[layer_data['Model'] == 'GPT2 WordLevel']['diff_base_emb']
        diff_base_emb_viejo = layer_data[layer_data['Model'] == 'GPT2 base']['diff_base_emb']
        
        # T-test to compare means of the two models
        test_modelos = stats.ttest_ind(diff_base_emb_WL, diff_base_emb_viejo, alternative='greater')
        
        # Prepare human bias data
        diff_base_humanos = pd.concat([
            layer_data[layer_data['meaningID'] == 1]['diff_base1'],
            layer_data[layer_data['meaningID'] == 2]['diff_base2']
        ])
        
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
    for layer in data['layer'].unique():
        layer_data = data[data['layer'] == layer]
        
        plt.figure(figsize=(30, 18))
        sns.kdeplot(data=layer_data[layer_data['Model'] == 'GPT2 base']['diff_base_emb'], 
                    fill=True, color="red", label="base", warn_singular=False)
        sns.kdeplot(data=layer_data[layer_data['Model'] == 'GPT2 WordLevel']['diff_base_emb'], 
                    fill=True, color="blue", label="Word Level",warn_singular=False)
        plt.xlabel("Difference between base and meaning embeddings")
        plt.ylabel("Density")
        plt.title(f"Density Plots - Layer {layer}")
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/density_plots/layer_{layer}.png", dpi=300)
        plt.savefig(f"{output_dir}/density_plots/layer_{layer}.svg", dpi=300)
        plt.close()

# Create density plots for each layer
create_density_plots(df)