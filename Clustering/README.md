# Pokemon Clustering

## Apply Kohonen layers to cluster pokemons.

### [**Code**](/Clustering/)

### Procedure 

1. A dataset containing Pokémon and their characteristics was obtained. Uninformative columns were dropped, and string-type features were transformed using Label Encoding. The "Abilities" column was transformed using TF-IDF.

2. A Random Forest classifier was trained to fill in missing values.

3. The elbow method was applied to determine the number of clusters.

4. Kohonen layers were created using the MiniSom library, and clustering was performed using neural networks.

