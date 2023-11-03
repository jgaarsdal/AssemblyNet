import torch
import torch.nn.functional as F

def construct_sord_labels(vectors, kernel="cosine_distance"):
    
    # Assert shape 2 dimensional e.g. is N x 3. Where N is the number of vectors
    assert vectors.ndim == 2
    N, _ = vectors.shape

    label_weights = torch.zeros((N,N))

    for idx1 in range(N):
        for idx2 in range(N):
            
            # Compute metric difference between vectors using a predetermined kernel (phi in the SORD paper) 
            # Only criteria is that a lower number represents being more "similar"
            if kernel == "cosine_distance":
                metric_diff = 1 - F.cosine_similarity(vectors[idx1],vectors[idx2], eps=1e-08, dim=-1)
            elif kernel == "euclidean_distance":
                metric_diff = (vectors[idx1]-vectors[idx2]).pow(2).sum().sqrt()
            elif kernel == "manhattan_distance":
                metric_diff = (vectors[idx1]-vectors[idx2]).abs().sum()

            label_weights[idx1,idx2] = metric_diff
        
    
    print("SORD labels kernel:")
    print(kernel)
    print("SORD label weights before normalization:")
    print(label_weights)

    # Eq. 1 in the SORD paper. Normalizes output to sum to 1
    label_weights = F.softmax(-label_weights, dim=-1)

    print("SORD label weights after normalization:")
    print(label_weights)
    print()

    return label_weights



if __name__ == "__main__":
    vectors = torch.Tensor([[0,0,1],[0,1,0],[0,1,1], [0, 0.2, 0.8]])

    weights = construct_sord_labels(vectors, "cosine_distance")
    construct_sord_labels(vectors, "euclidean_distance")
    construct_sord_labels(vectors, "manhattan_distance")
            
    target = 2
    predicted = 3
    print(weights[target, predicted])
