import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Find the optimal number of clusters
def optimal_number_of_clusters(wcss):
    # Defube x1, y1, x2, y2 to be the first and last coordinates of the elbow curve
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss) - 1]


    distances = []
    for i in range(len(wcss)):
        x0 = i + 2
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(numerator / denominator)

    return distances.index(max(distances)) + 2

def kmeans(data, name, fd_name, save=False):

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

    # Find the optimal number of clusters using the elbow method
    wcss = []
    for i in range(1, 21):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init="auto", max_iter=300)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    n = optimal_number_of_clusters(wcss)
    print(f"Optimal number of clusters: {n}")

    # Train the model
    kmeans = KMeans(n_clusters=n, init='k-means++', random_state=42, n_init="auto", max_iter=300)
    kmeans.fit(scaled_data)

    # Add the cluster labels to the dataframes
    data.loc[:, 'cluster'] = kmeans.labels_

    if save:
        data.to_csv(f'data/{name}_clusters.csv')

    # # Plot the clusters on a 3D graph
    # fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # for c in data['cluster'].unique():
    #     cluster = data[data['cluster'] == c]
    #     ax.scatter(cluster[f'{name}_rate'], cluster[f'{name}_transaction_count'], cluster[f'{name}_total_amount'], label=f'Cluster {c}')

    # ax.set_xlabel(f'{name} rate')
    # ax.set_ylabel(f'{name} transaction count')
    # ax.set_zlabel(f'{name} total amount')

    # # order the legend by cluster number
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # by_label = dict(sorted(by_label.items(), key=lambda item: int(item[0].split(" ")[1])))
    # ax.legend(by_label.values(), by_label.keys())
    # # plt.savefig(f'{name}_clusters3d.png')
    # plt.show()
    # return

    # plot the clusters on a 2D graph
    plt.figure(figsize=(12, 7))
    for c in data['cluster'].unique():
        cluster = data[data['cluster'] == c]
        plt.scatter(cluster[f'{name}_rate'], cluster[f'{name}_transaction_count'], label=f'Cluster {c}')

    plt.xlabel(f'{name} rate')
    plt.ylabel(f'{name} transaction count')
    plt.legend()
    if save: plt.savefig(f'{fd_name}/cluster/{name}_clusters2d.png')
    plt.show()

    # # make 3 2d subplot, one for each pair of features
    # fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    # for c in data['cluster'].unique():
    #     cluster = data[data['cluster'] == c]
    #     ax[0].scatter(cluster[f'{name}_rate'], cluster[f'{name}_transaction_count'], label=f'Cluster {c}')
    #     ax[1].scatter(cluster[f'{name}_rate'], cluster[f'{name}_total_amount'], label=f'Cluster {c}')
    #     ax[2].scatter(cluster[f'{name}_transaction_count'], cluster[f'{name}_total_amount'], label=f'Cluster {c}')

    # ax[0].set_xlabel(f'{name} rate')
    # ax[0].set_ylabel(f'{name} transaction count')
    # ax[0].legend()
    # ax[1].set_xlabel(f'{name} rate')
    # ax[1].set_ylabel(f'{name} total amount')
    # ax[1].legend()
    # ax[2].set_xlabel(f'{name} transaction count')
    # ax[2].set_ylabel(f'{name} total amount')
    # ax[2].legend()
    # if save: plt.savefig(f'{fd_name}/cluster/{name}_subplots_clusters2d.png')
    # plt.show()

def line_plot_by_state(data, name, fd_name, save=False):
    # make 3 subplots, one for each feature, let enough space for the xlabel
    fig, ax = plt.subplots(3, 1, figsize=(30, 15))
    fig.suptitle(f'{name} by hour', fontsize=16)

    ax[0].plot(data.index, data[f'{name}_rate'])
    ax[1].plot(data.index, data[f'{name}_transaction_count'])
    ax[2].plot(data.index, data[f'{name}_total_amount'])

    ax[0].set_ylabel(f'{name} rate')
    ax[1].set_ylabel(f'{name} transaction count')
    ax[2].set_ylabel(f'{name} total amount')
    ax[2].set_xlabel('Hour')

    # plot an horizontal line for the mean of each feature
    # ax[0].axhline(y=data[f'{name}_rate'].mean(), color='r', linestyle='--', label='mean')
    # ax[1].axhline(y=data[f'{name}_transaction_count'].mean(), color='r', linestyle='--', label='mean')
    # ax[2].axhline(y=data[f'{name}_total_amount'].mean(), color='r', linestyle='--', label='mean')

    # plot an horizontal line for the 25th, 50th, 75th and 90th percentile of each feature
    ax[0].axhline(y=data[f'{name}_rate'].quantile(0.25), color='y', linestyle='--', label='25th')
    ax[0].axhline(y=data[f'{name}_rate'].quantile(0.5), color='r', linestyle='--', label='50th')
    ax[0].axhline(y=data[f'{name}_rate'].quantile(0.75), color='g', linestyle='--', label='75th')
    ax[0].axhline(y=data[f'{name}_rate'].quantile(0.9), color='m', linestyle='--', label='90th')
    ax[1].axhline(y=data[f'{name}_transaction_count'].quantile(0.25), color='y', linestyle='--', label='25th')
    ax[1].axhline(y=data[f'{name}_transaction_count'].quantile(0.5), color='r', linestyle='--', label='50th')
    ax[1].axhline(y=data[f'{name}_transaction_count'].quantile(0.75), color='g', linestyle='--', label='75th')
    ax[1].axhline(y=data[f'{name}_transaction_count'].quantile(0.9), color='m', linestyle='--', label='90th')
    ax[2].axhline(y=data[f'{name}_total_amount'].quantile(0.25), color='y', linestyle='--', label='25th')
    ax[2].axhline(y=data[f'{name}_total_amount'].quantile(0.5), color='r', linestyle='--', label='50th')
    ax[2].axhline(y=data[f'{name}_total_amount'].quantile(0.75), color='g', linestyle='--', label='75th')
    ax[2].axhline(y=data[f'{name}_total_amount'].quantile(0.9), color='m', linestyle='--', label='90th')

    # add a legend to explain the horizontal lines colors
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()


    # plot a vertical line for each day
    xticks_pos = np.arange(0, len(data)+1, 24)
    xticks_labels = [f'D{i+1}' for i in range(len(xticks_pos))]
    ax[0].set_xticks(xticks_pos)
    ax[0].set_xticklabels(xticks_labels)
    ax[1].set_xticks(xticks_pos)
    ax[1].set_xticklabels(xticks_labels)
    ax[2].set_xticks(xticks_pos)
    ax[2].set_xticklabels(xticks_labels)

    for xtick in xticks_pos:
        ax[0].axvline(x=xtick, color='k', linestyle='--', linewidth=0.5)
        ax[1].axvline(x=xtick, color='k', linestyle='--', linewidth=0.5)
        ax[2].axvline(x=xtick, color='k', linestyle='--', linewidth=0.5)

    if save: plt.savefig(f'{fd_name}/state_{name}_perc.png')
    # plt.show()

def line_plot_by_feature(data, name, fd_name, save=False):
    fig, ax = plt.subplots(3, 1, figsize=(30, 15))
    fig.suptitle(f'{name} by hour', fontsize=16)

    ax[0].plot(data.index, data[f'paid_{name}'])
    ax[1].plot(data.index, data[f'unpaid_{name}'])
    ax[2].plot(data.index, data[f'abandoned_{name}'])

    ax[0].set_ylabel(f'paid')
    ax[1].set_ylabel(f'unpaid')
    ax[2].set_ylabel(f'abandoned')
    ax[2].set_xlabel('Hour')

    # plot an horizontal line for the mean of each feature
    # ax[0].axhline(y=data[f'paid_{name}'].mean(), color='r', linestyle='--', label='mean')
    # ax[1].axhline(y=data[f'unpaid_{name}'].mean(), color='r', linestyle='--', label='mean')
    # ax[2].axhline(y=data[f'abandoned_{name}'].mean(), color='r', linestyle='--', label='mean')

    # plot an horizontal line for the 75th and 90th percentile of each feature
    ax[0].axhline(y=data[f'paid_{name}'].quantile(0.25), color='y', linestyle='--', label='25th')
    ax[0].axhline(y=data[f'paid_{name}'].quantile(0.5), color='r', linestyle='--', label='50th')
    ax[0].axhline(y=data[f'paid_{name}'].quantile(0.75), color='g', linestyle='--', label='75th')
    ax[0].axhline(y=data[f'paid_{name}'].quantile(0.9), color='m', linestyle='--', label='90th')
    ax[1].axhline(y=data[f'unpaid_{name}'].quantile(0.25), color='y', linestyle='--', label='25th')
    ax[1].axhline(y=data[f'unpaid_{name}'].quantile(0.5), color='r', linestyle='--', label='50th')
    ax[1].axhline(y=data[f'unpaid_{name}'].quantile(0.75), color='g', linestyle='--', label='75th')
    ax[1].axhline(y=data[f'unpaid_{name}'].quantile(0.9), color='m', linestyle='--', label='90th')
    ax[2].axhline(y=data[f'abandoned_{name}'].quantile(0.25), color='y', linestyle='--', label='25th')
    ax[2].axhline(y=data[f'abandoned_{name}'].quantile(0.5), color='r', linestyle='--', label='50th')
    ax[2].axhline(y=data[f'abandoned_{name}'].quantile(0.75), color='g', linestyle='--', label='75th')
    ax[2].axhline(y=data[f'abandoned_{name}'].quantile(0.9), color='m', linestyle='--', label='90th')

    # add a legend to explain the horizontal lines colors
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    # plot a vertical line for each day
    xticks_pos = np.arange(0, len(data)+1, 24)
    xticks_labels = [f'D{i+1}' for i in range(len(xticks_pos))]
    ax[0].set_xticks(xticks_pos)
    ax[0].set_xticklabels(xticks_labels)
    ax[1].set_xticks(xticks_pos)
    ax[1].set_xticklabels(xticks_labels)
    ax[2].set_xticks(xticks_pos)
    ax[2].set_xticklabels(xticks_labels)


    for xtick in xticks_pos:
        ax[0].axvline(x=xtick, color='k', linestyle='--', linewidth=0.5)
        ax[1].axvline(x=xtick, color='k', linestyle='--', linewidth=0.5)
        ax[2].axvline(x=xtick, color='k', linestyle='--', linewidth=0.5)

    if save: plt.savefig(f'{fd_name}/feature_{name}_perc.png')
    # plt.show()

def stats(data_fn, fd_name):
    
    # Load data into a pandas dataframe
    data = pd.read_csv(data_fn)

    # keep only paid_rate, unpaid_rate and abandoned_rate
    # data = data[['paid_rate', 'unpaid_rate', 'abandoned_rate']]

    # # export columns by "state"
    # data_paid = data[['paid_rate', 'paid_transaction_count', 'paid_total_amount']]
    # data_unpaid = data[['unpaid_rate', 'unpaid_transaction_count', 'unpaid_total_amount']]
    # data_abandoned = data[['abandoned_rate', 'abandoned_transaction_count', 'abandoned_total_amount']]
    data_paid = data[['paid_rate', 'paid_transaction_count']]
    data_unpaid = data[['unpaid_rate', 'unpaid_transaction_count']]
    data_abandoned = data[['abandoned_rate', 'abandoned_transaction_count']]

    saveit = True
    kmeans(data_paid, 'paid', fd_name, save=saveit)
    kmeans(data_unpaid, 'unpaid', fd_name, save=saveit)
    kmeans(data_abandoned, 'abandoned', fd_name, save=saveit)


    # line_plot_by_state(data, 'paid', fd_name, save=saveit)
    # line_plot_by_state(data, 'unpaid', fd_name, save=saveit)
    # line_plot_by_state(data, 'abandoned', fd_name, save=saveit)

    # line_plot_by_feature(data, 'rate', fd_name, save=saveit)
    # line_plot_by_feature(data, 'transaction_count', fd_name, save=saveit)
    # line_plot_by_feature(data, 'total_amount', fd_name, save=saveit)

# data_fn = '../database/data/data_ogone.csv'
# fd_name = 'plots/ogone/year'

data_fn = '../database/data/month/data_ogone.csv'
fd_name = 'plots/ogone/month'

# # data_fn = '../database/data/month/data_adyen.csv'
# # fd_name = 'plots/adyen/month'

# stats(data_fn, fd_name)
# exit()



def plot_data_by_cluster(data, types, fd_name, save=False):
    # create a color list that contains 10 different colors
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']

    plt.figure(figsize=(30, 5)) 
    x = data['index']
    y = data[f'{types}_rate']
    current_cluster = data['cluster'][0]

    for i in range(1, len(data)):
        if data['cluster'][i] != current_cluster:
            plt.plot(x[:i], y[:i], color=colors[current_cluster], label=f'Cluster {current_cluster}')
            x = x[i-1:]
            y = y[i-1:]
            current_cluster = data['cluster'][i]

    # Plot the last part of the line
    plt.plot(x, y, color=colors[current_cluster], label=f'Cluster {current_cluster}')

    plt.xlabel('Index')
    plt.ylabel(f'{types} Rate')
    plt.title('Line Chart with Cluster Color')

    # plot a vertical line for each day
    xticks_pos = np.arange(0, len(data)+1, 24)
    xticks_labels = [f'D{i+1}' for i in range(len(xticks_pos))]
    plt.xticks(xticks_pos, xticks_labels)
    
    for xtick in xticks_pos:
        plt.axvline(x=xtick, color='k', linestyle='--', linewidth=0.5)
    

    # Display a single legend for each cluster
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items(), key=lambda item: int(item[0].split(" ")[1])))
    plt.legend(by_label.values(), by_label.keys())

    if save: plt.savefig(f'{fd_name}/{types}_rate_cluster_color.png')
    plt.show()


def plotting(data, idx, types, fd_name, save=False):
    plt.figure(figsize=(10, 6))  # Ajustez la taille du graphique selon vos préférences
    x = data['index']
    y = data[f'{types}_rate']
    # the current cluster is the one with the index idx
    current_cluster = data['cluster'][idx]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']


    # print number of elements in cluster
    print(f'cluster {idx} has {len(data[data["cluster"] == idx])} elements')

    for i in range(1, len(data)):
        if data['cluster'][i] != current_cluster: # if the cluster changes
            if current_cluster == idx: # if the cluster is the one we want to plot
                plt.plot(x[:i], y[:i], color=colors[current_cluster], label=f'Cluster {current_cluster}')
            else:
                plt.plot(x[:i], y[:i], color='w', label=f'Cluster {current_cluster}')
            x = x[i-1:]
            y = y[i-1:]
            current_cluster = data['cluster'][i] # update the current cluster by the new one

    # Tracer la dernière partie de la ligne
    if current_cluster == idx:
        print("enter")
        plt.plot(x, y, color=colors[current_cluster], label=f'Cluster {current_cluster}')
    else:
        plt.plot(x, y, color='w', label=f'Cluster {current_cluster}')

    # Personnaliser le graphique
    plt.xlabel('Index')
    plt.ylabel(f'{types} Rate')
    plt.title('Line Chart with Cluster Color')

    # Afficher une seule légende pour chaque cluster
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # order the legend by cluster number
    by_label = dict(sorted(by_label.items(), key=lambda item: int(item[0].split(" ")[1])))
    plt.legend(by_label.values(), by_label.keys())

    # Afficher le graphique
    if save: plt.savefig(f'{fd_name}/precise/{types}_rate_cluster_{idx}.png')
    plt.show()



# display the values of the 7th cluster
# print(data[data['cluster'] == 2])
# print(data[data['cluster'] == 5])
# print(data[data['cluster'] == 6])

# for i in range(5, len(clusters)):
#     plotting(data, i)


for types in ['paid', 'unpaid', 'abandoned']:

    data_fn = f'./data/{types}_clusters.csv'
    fd_name = f'plots/clusters/{types}'

    data = pd.read_csv(data_fn)


    plot_data_by_cluster(data, types, fd_name, save=True)

    clusters = data['cluster'].unique()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']


    # for each cluster make the subplots on a bar chart the number of elements for each hour (the index needs to be mod 24 to get the hour)
    # compute the right number of row and column for the subplots, to have a square shape like
    rows = int(np.sqrt(len(clusters)))
    cols = int(np.ceil(len(clusters) / rows))

    fig, ax = plt.subplots(rows, cols, figsize=(20, 10))
    fig.suptitle(f'Number of elements by hour for each cluster of {types}', fontsize=16)

    for i in range(len(clusters)):
        cluster = data[data['cluster'] == i]
        cluster = cluster.reset_index(drop=True)
        cluster['index'] = cluster['index'] % 24
        ax[i // cols, i % cols].bar(cluster['index'], cluster[f'{types}_rate'], color=colors[i])
        ax[i // cols, i % cols].set_title(f'Cluster {i}')
        # set the xticks to be the hours from 0 to 23
        ax[i // cols, i % cols].set_xticks(np.arange(0, 24))


    plt.savefig(f'{fd_name}/{types}nb_elements_by_cluster.png')
    plt.show()
    





    # rate_means = [data[data['cluster'] == i][f'{types}_rate'].mean() for i in range(len(clusters))]
    # transaction_count_means = [data[data['cluster'] == i][f'{types}_transaction_count'].mean() for i in range(len(clusters))]
    # total_amount_means = [data[data['cluster'] == i][f'{types}_total_amount'].mean() for i in range(len(clusters))]
    # # make 3 subplots for each features and plot the values in a bar chart
    # fig, ax = plt.subplots(3, 1, figsize=(20, 10))
    # fig.suptitle(f'Mean {types} values by cluster', fontsize=16)

    # ax[0].bar(clusters, rate_means)
    # ax[1].bar(clusters, transaction_count_means)
    # ax[2].bar(clusters, total_amount_means)

    # ax[0].set_ylabel(f'{types} rate')
    # ax[1].set_ylabel(f'{types} transaction count')
    # ax[2].set_ylabel(f'{types} total amount')
    # ax[2].set_xlabel('Cluster')

    # plt.savefig(f'{fd_name}/mean_{types}_values_by_cluster.png')
    # plt.show()

    # for i in range(len(clusters)):
    #     # plotting(data, i, types, fd_name, save=True)

    #     cluster = data[data['cluster'] == i]
        
    #     # print(f"Cluster {i}:")
    #     # print(f"Mean {types} rate: {cluster[f'{types}_rate'].mean():.2f}")
    #     # print(f"Mean {types} transaction count: {cluster[f'{types}_transaction_count'].mean():.2f}")
    #     # print(f"Mean {types} total amount: {cluster[f'{types}_total_amount'].mean():.2f}")




