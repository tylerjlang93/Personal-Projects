import matplotlib.image as img
from PIL import Image
import numpy as np
import pandas as pd
import random as r
from scipy.spatial.distance import cdist
import random
from collections import defaultdict
from sklearn.cluster import KMeans

def assign_pts(m,k,df,centroids):
    D = np.zeros((m,k))
    for i in range(k):
        temp = np.linalg.norm(df - centroids[i,:],axis=1)
        D[:,i] += temp
    labels = np.argmin(D,axis=1)
    return labels



def new_centroids(n,k,df,labels):
    cen = np.zeros((k,n))
    for i in range(k):
        cen[i,:] += np.mean(df[labels==i],axis=0)
    return cen





def kmeans_tjl(k, df):
    k_orig = k
    m = df.shape[0]
    mins = np.min(df,axis=0)
    maxs = np.max(df,axis=0)

    k_sizing = True
    while k_sizing:
        old_centroids = np.zeros((k,3))
        for i in range(df.shape[1]):
            old_centroids[:,i] += random.choices(range(mins[i],maxs[i]+1),k=k)
        labels = assign_pts(m,k,df,old_centroids)

        if list(pd.Series(labels).sort_values().unique()) == list(range(k)):
            new_cens = new_centroids(3,k,df,labels)
            k_sizing = False
        else:
            k -= 1

    repetitions=0
    while repetitions == 0:
        while abs(np.linalg.norm(new_cens - old_centroids)) > 0:
            old_centroids = new_cens.copy()
            labels = assign_pts(m,k,df,old_centroids)
            #if list(pd.Series(labels).sort_values().unique()) == list(range(k)):
            new_cens = new_centroids(3,k,df,labels)
            #else:
                #kmeans_tjl(k-1,df)
            repetitions+=1
    print(f'Original number of clusters called: {k_orig}')
    print(f'Number of Clusters: {k}')
    print(f'Iterations: {repetitions} \n')
    print(f'Centroids: \n {new_cens} \n')
    #for i in range(k):
        #print(f'Number of pixels in cluster {i}: {(labels == i).sum()}')
    return labels, new_cens









def show_separate_layers(df,shape,labels):
    temp = df.copy()
    for i in pd.Series(labels).sort_values().unique():
        temp[labels!=i,:] = (0,0,0)
        df_new = temp.reshape(shape)
        im = Image.fromarray(df_new, 'RGB')
        display(im)
        temp = df.copy()


def compress_img(df,shape,labels,centroids):
    temp = df.copy()
    for i in pd.Series(labels).sort_values().unique():
        temp[labels==i,:] = centroids[i]
    temp = temp.reshape(shape)
    temp_img = Image.fromarray(temp,'RGB')
    display(temp_img.resize((int(temp_img.size[0]/1.5),int(temp_img.size[1]/1.5))))





def assign_pts_manh(m,k,df,centroids):
    D = np.zeros((m,k))
    for i in range(k):
        temp = cdist(df, centroids[i,:].reshape((1,3)), metric='cityblock').reshape((df.shape[0],))
        D[:,i] += temp
    labels = np.argmin(D,axis=1)
    return labels

def new_centroids_manh(n,k,df,labels):
    cen = np.zeros((k,n))
    for i in range(k):
        cen[i,:] += np.median(df[labels==i],axis=0)
    return cen




def kmeans_manhattan_tjl(k, df):
    k_orig = k
    m = df.shape[0]

    mins = np.min(df,axis=0)
    maxs = np.max(df,axis=0)

    k_sizing = True
    while k_sizing:
        old_centroids = np.zeros((k,3))
        for i in range(df.shape[1]):
            old_centroids[:,i] += random.choices(range(mins[i],maxs[i]+1),k=k)
        labels = assign_pts_manh(m,k,df,old_centroids)

        if list(pd.Series(labels).sort_values().unique()) == list(range(k)):
            new_cens = new_centroids_manh(3,k,df,labels)
            k_sizing = False
        else:
            k -= 1

    repetitions=0
    while repetitions == 0:
        while abs(np.linalg.norm(new_cens - old_centroids)) > 0:
            old_centroids = new_cens.copy()
            labels = assign_pts_manh(m,k,df,old_centroids)
            new_cens = new_centroids_manh(3,k,df,labels)
            repetitions+=1

    print(f'Original number of clusters called: {k_orig}')
    print(f'Number of Clusters: {k}')
    print(f'Iterations: {repetitions} \n')
    print(f'Centroids: \n {new_cens}')
    #for i in range(k):
        #print(f'Number of pixels in cluster {i}: {(labels == i).sum()}')
    return labels, new_cens


def pre_processing(nodes,edges):
    e = edges.drop_duplicates(['ind1','ind2'])
    m = pd.DataFrame(np.sort(e[['ind1','ind2']], axis=1), index = e.index).duplicated()
    # This code for removing symmetric duplicates from Stack Overflow https://stackoverflow.com/questions/58450926/pandas-removing-symmetrical-duplicates-in-two-rows-from-a-dataframe
    e = e[~m]

    n = nodes.copy()
    inds_to_remove = []
    for i in range(len(nodes)):
        if (nodes.iloc[i,0] in e['ind1']) or (nodes.iloc[i,0] in e['ind2']):
            continue
        else:
            inds_to_remove.append(nodes.iloc[i,0])

    inds = sorted(list(set(list(e['ind1']) + list(e['ind2']))))

    n_link = nodes[nodes['ind_ref'].isin(inds)]
    n_link = n_link.reset_index(drop=True).reset_index().rename({'index':'new_ind'},axis=1)
    re_indexer_df = n_link[['new_ind','ind_ref']]
    re_indexer_df
    re_indexer = defaultdict()
    for i in range(len(re_indexer_df)):
        re_indexer[re_indexer_df['ind_ref'][i]] = re_indexer_df['new_ind'][i]

    e['ind1_new'] = e['ind1'].map(re_indexer)
    e['ind2_new'] = e['ind2'].map(re_indexer)
    e = e.drop(['ind1','ind2'],axis=1)
    ints = list(e[e['ind1_new']==e['ind2_new']].index)
    e = e[~e.index.isin(ints)]
    return e, n_link

def spectral_setup(new_adj_df,nodes_processed):
    E = np.array(new_adj_df)
    size_A = nodes_processed.shape[0]
    A = np.empty((size_A,size_A))

    for row, col in E:
        A[row][col] = 1
        A[col][row] = 1

    D_vals = np.sum(A,axis=1)
    D = np.diag(D_vals)
    L = D-A

    v, x= np.linalg.eig(L)
    idx_sorted = (np.argsort(v))
    return v,x,idx_sorted


def spectral_setup_alternate(new_adj_df,nodes_processed):
    E = np.array(new_adj_df)
    size_A = nodes_processed.shape[0]
    A = np.empty((size_A,size_A))

    for row, col in E:
        A[row][col] = 1
        A[col][row] = 1
    A = np.matrix(A)

    # Code for calculating D and L directly referenced from sample code given to us.

    D = np.diag(1/np.sqrt(np.sum(A, axis=1)).A1)
    L = D @ A @ D
    L = np.array(L)

    v, x= np.linalg.eig(L)
    idx_sorted = (np.argsort(v))[::-1]
    return v,x,idx_sorted



def spectral_kmeans(k,x,idx_sorted,nodes):
    Z = x[:, idx_sorted[:k]].real
    Z = Z/np.repeat(np.sqrt(np.sum(Z*Z, axis=1).reshape(-1,1)), k, axis=1)
    # This step for normalizing Z was taken from sample code given to us
    kmeans = KMeans(n_clusters=k).fit(Z)
    labs = kmeans.labels_
    labs_df = pd.DataFrame(labs).reset_index().rename({0:'label'},axis=1)
    node_with_label = nodes.merge(labs_df,left_on='new_ind',right_on='index')
    label_polit = node_with_label[['political_label','label']]
    counts = label_polit.value_counts().reset_index().rename({0:'count'},axis=1)
    highest_count_label = counts.groupby('label')['count'].max().reset_index()
    c_m = counts.merge(highest_count_label,on='label')
    c_m = c_m[c_m['count_x']==c_m['count_y']]
    total_label_counts = counts.groupby('label')['count'].sum().reset_index().rename({'count':'totals'},axis=1)
    final = counts.merge(total_label_counts, on='label').merge(c_m[['label','political_label']])
    final['mismatch_rate'] = (final['totals'] - final['count'])/final['totals']
    final['mismatches'] = final['totals']-final['count']
    mr=final['mismatches'].sum()/final['totals'].sum()
    return final, mr
