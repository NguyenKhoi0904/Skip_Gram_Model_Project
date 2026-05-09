from sklearn.decomposition  import PCA

import pandas as pd
from matplotlib import pyplot as plt


def Solve():
    embedding_matrix = pd.read_csv('w1.csv', header= None)
    pca = PCA(n_components= 2)
    reduce_embedding = pca.fit_transform(embedding_matrix)
    
    plt.figure(figsize=(10,8))
    plt.scatter(reduce_embedding[:,0], reduce_embedding[:,1], alpha= 0.6)
    plt.title('Trực quan hóa ma trận nhúng')
    plt.xlabel('Tính năng 1')
    plt.ylabel('Tính năng 2')
    plt.savefig('embedding.png', dpi = 300, bbox_inches = 'tight')
    

def main():
    Solve()
    
            

if __name__ == "__main__":
    main()