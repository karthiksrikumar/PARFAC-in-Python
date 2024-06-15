import pandas as pd
import numpy as np
from tensorly import cp_to_tensor
from tensorly.decomposition import parafac

df = pd.read_csv('numbers.csv')

entities = df['Entity'].values
training_computations = df['Training computation (petaFLOP)'].values * 0.8

unique_domains = df['Domain'].unique()
domain_map = {domain: i for i, domain in enumerate(unique_domains)}
domains_numerical = df['Domain'].map(domain_map).values
domains = domains_numerical.astype(float) * 0.2

unique_entities = np.unique(entities)
entity_ids = {entity: i for i, entity in enumerate(unique_entities)}
entities_numerical = [entity_ids[entity] for entity in entities]

tensor = np.zeros((len(unique_entities), len(unique_domains)))
tensor[np.array(entities_numerical)[:, None], domains_numerical] = training_computations + domains

rank = 5
weights, factors = parafac(tensor, rank=rank, init='random')

print("Component Vectors (Factors):")
print("Entity Components:")
for i, factor in enumerate(factors[0].T):
    print(f"Component {i+1}: {', '.join(f'{val:.2f}' for val in factor)}")
print("\nDomain Components:")
for i, factor in enumerate(factors[1].T):
    print(f"Component {i+1}: {', '.join(f'{val:.2f}' for val in factor)}")
print("\nWeights:")
print(", ".join(f"{weight:.2f}" for weight in weights))