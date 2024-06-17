# PARFAC

This project is a Python script that performs tensor factorization, specifically the Parallel Factor Analysis (PARAFAC) decomposition, on a dataset containing entities, domains, and training computations. The PARAFAC decomposition is a powerful technique in tensor analysis that aims to decompose a tensor into a sum of component rank-one tensors. In this context, the script constructs a tensor from the dataset, representing the relationships between entities, domains, and training computations, and then decomposes it using the PARAFAC algorithm.

The tensor factorization process involves representing the original tensor as a sum of rank-one tensors, each weighted by a scalar value. The rank-one tensors are the outer products of vectors, known as component vectors or factors. These component vectors capture the latent patterns or components present in the data across different modes (entities, domains, and training computations in this case).

By performing the PARAFAC decomposition, the script aims to identify the underlying components that contribute to the observed training computations for each entity-domain combination. The component vectors represent the influence or importance of each component for entities and domains, respectively. The weights associated with each rank-one tensor indicate the relative contribution of that component to the overall tensor.

The script's output includes the component vectors (factors) for entities and domains, as well as the weights for each rank-one tensor. These components and weights can be analyzed and interpreted to gain insights into the relationships and patterns within the dataset. For example, domain components with high values for certain components may indicate that those components are particularly influential or important for specific domains. Similarly, entity components can reveal patterns or similarities among entities based on their component values.

The PARAFAC decomposition is a powerful tool in tensor analysis and has applications in various fields, including signal processing, chemometrics, psychometrics, and machine learning. By leveraging this technique, the script aims to uncover latent patterns and relationships within the provided dataset, potentially enabling further analysis, modeling, or decision-making processes.

To run the script, users need to have the Pandas, NumPy, and TensorLy libraries installed. The script reads the dataset from a CSV file named "numbers.csv" and expects the file to contain columns for "Entity," "Training computation (petaFLOP)," and "Domain." After preprocessing the data and constructing the tensor, the script performs the PARAFAC decomposition with a specified rank and outputs the component vectors and weights.

We can apply this to one primary domain: 
  Recommender Systems: PARAFAC can be used to build recommendation models for products, movies, or music by analyzing the tensor formed by users, items, and contextual information (e.g., time, location). The component vectors can capture latent preferences or characteristics of users and items, aiding in personalized recommendations.
