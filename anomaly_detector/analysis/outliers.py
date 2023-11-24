import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

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

    # Find the optimal number of clusters using the elbow type
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

def line_plot_by_state(data, name, fd_name, save=False, percentiles=False):
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
    ax[0].axhline(y=data[f'{name}_rate'].quantile(0.5), color='r', linestyle='--', label='50th')
    ax[1].axhline(y=data[f'{name}_transaction_count'].quantile(0.5), color='r', linestyle='--', label='50th')
    ax[2].axhline(y=data[f'{name}_total_amount'].quantile(0.5), color='r', linestyle='--', label='50th')
    if percentiles:
        ax[0].axhline(y=data[f'{name}_rate'].quantile(0.25), color='y', linestyle='--', label='25th')
        ax[0].axhline(y=data[f'{name}_rate'].quantile(0.75), color='g', linestyle='--', label='75th')
        ax[0].axhline(y=data[f'{name}_rate'].quantile(0.9), color='m', linestyle='--', label='90th')
        ax[1].axhline(y=data[f'{name}_transaction_count'].quantile(0.25), color='y', linestyle='--', label='25th')
        ax[1].axhline(y=data[f'{name}_transaction_count'].quantile(0.75), color='g', linestyle='--', label='75th')
        ax[1].axhline(y=data[f'{name}_transaction_count'].quantile(0.9), color='m', linestyle='--', label='90th')
        ax[2].axhline(y=data[f'{name}_total_amount'].quantile(0.25), color='y', linestyle='--', label='25th')
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

def line_plot_by_feature(data, name, fd_name, save=False, percentiles=False):
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
    ax[0].axhline(y=data[f'paid_{name}'].quantile(0.5), color='r', linestyle='--', label='50th')
    ax[1].axhline(y=data[f'unpaid_{name}'].quantile(0.5), color='r', linestyle='--', label='50th')
    ax[2].axhline(y=data[f'abandoned_{name}'].quantile(0.5), color='r', linestyle='--', label='50th')

    if percentiles:
        ax[0].axhline(y=data[f'paid_{name}'].quantile(0.25), color='y', linestyle='--', label='25th')
        ax[0].axhline(y=data[f'paid_{name}'].quantile(0.75), color='g', linestyle='--', label='75th')
        ax[0].axhline(y=data[f'paid_{name}'].quantile(0.9), color='m', linestyle='--', label='90th')
        ax[1].axhline(y=data[f'unpaid_{name}'].quantile(0.25), color='y', linestyle='--', label='25th')
        ax[1].axhline(y=data[f'unpaid_{name}'].quantile(0.75), color='g', linestyle='--', label='75th')
        ax[1].axhline(y=data[f'unpaid_{name}'].quantile(0.9), color='m', linestyle='--', label='90th')
        ax[2].axhline(y=data[f'abandoned_{name}'].quantile(0.25), color='y', linestyle='--', label='25th')
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
    # kmeans(data_paid, 'paid', fd_name, save=saveit)
    # kmeans(data_unpaid, 'unpaid', fd_name, save=saveit)
    # kmeans(data_abandoned, 'abandoned', fd_name, save=saveit)


    # line_plot_by_state(data, 'paid', fd_name, save=saveit)
    # line_plot_by_state(data, 'unpaid', fd_name, save=saveit)
    # line_plot_by_state(data, 'abandoned', fd_name, save=saveit)

    line_plot_by_feature(data, 'rate', fd_name, save=saveit)
    line_plot_by_feature(data, 'transaction_count', fd_name, save=saveit)
    line_plot_by_feature(data, 'total_amount', fd_name, save=saveit)


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


def clustered():

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


def stats(data, data_inf, data_sup):
    print(f"Lenght of data: {len(data)}")
    print(f"Lenght of data_inf: {len(data_inf)}")
    print(f"Lenght of data_sup: {len(data_sup)}")

    print()

    print(f"Mean all", data['paid_rate'].mean())
    print(f"Mean sup", data_sup['paid_rate'].mean())
    print(f"Mean inf", data_inf['paid_rate'].mean())

    print(f"Std all", data['paid_rate'].std())
    print(f"Std sup", data_sup['paid_rate'].std())
    print(f"Std inf", data_inf['paid_rate'].std())

    print()

    print(f"Mean all", data['total_transaction'].mean())
    print(f"Mean sup", data_sup['total_transaction'].mean())
    print(f"Mean inf", data_inf['total_transaction'].mean())

    print(f"Std all", data['total_transaction'].std())
    print(f"Std sup", data_sup['total_transaction'].std())
    print(f"Std inf", data_inf['total_transaction'].std())

    input()

def stats2(data, data_inf, data_sup):
    for i in range(24):
        mean_rate = data[data.index.hour == i]['paid_rate'].mean()
        # mean_inf_rate = data_inf[data_inf.index.hour == i]['paid_rate'].mean()
        # mean_sup_rate = data_sup[data_sup.index.hour == i]['paid_rate'].mean()
        
        std_rate = data[data.index.hour == i]['paid_rate'].std()
        # std_inf_rate = data_inf[data_inf.index.hour == i]['paid_rate'].std()
        # std_sup_rate = data_sup[data_sup.index.hour == i]['paid_rate'].std()

        elems = data[(data.index.hour == i) & (data['paid_rate'] < (mean_rate - (2*std_rate)))]
        # elems = data[(data.index.hour == i) & (data['paid_rate'] < (mean_rate - (2*std_sup_rate)))]
        # elems_inf = data_inf[(data_inf.index.hour == i) & (data_inf['paid_rate'] < (mean_inf_rate - (2*std_inf_rate)))]
        # elems_sup = data_sup[(data_sup.index.hour == i) & (data_sup['paid_rate'] < (mean_sup_rate - (2*std_sup_rate)))]

        print(f"Hour {i}: {len(elems) / len(data[data.index.hour == i]) * 100:.2f}% ({len(elems)})")
        # print(f"Hour {i} inf: {len(elems_inf) / len(data_inf[data_inf.index.hour == i]) * 100:.2f}% ({len(elems_inf)})")
        # print(f"Hour {i} sup: {len(elems_sup) / len(data_sup[data_sup.index.hour == i]) * 100:.2f}% ({len(elems_sup)})")
        # print(elems)
        # print(elems_inf)
        # print(elems_sup)
        # input()

    exit()


data = pd.read_csv('../../database/data/data_ogone.csv')
data['total_transaction'] = data['paid_transaction_count'] + data['unpaid_transaction_count'] + data['abandoned_transaction_count']

data['timestamp'] = pd.date_range(start='2023-01-01 00:00:00', end='2023-09-30 23:59:59', freq='H')
data.set_index('timestamp', inplace=True)
# drop columns ['unpaid_rate', 'unpaid_transaction_count', 'unpaid_total_amount', 'abandoned_rate', 'abandoned_transaction_count', 'abandoned_total_amount']
data.drop(columns=['paid_total_amount', 'unpaid_rate', 'unpaid_transaction_count', 'unpaid_total_amount', 'abandoned_rate', 'abandoned_transaction_count', 'abandoned_total_amount'], inplace=True)

data = data.loc['2023-09'] # on garde que le mois de septembre 2023 pour vérifier qu'on fait correctement la prédiction

# print(data[data['total_transaction'] < data['paid_transaction_count']])
# input()

base_data = data.copy()

print(f"General mean : {data['paid_rate'].mean()}")
print(f"General std : {data['paid_rate'].std()}")
print(f"General mean : {data['total_transaction'].mean()}")
print(f"General std : {data['total_transaction'].std()}")
exit()
######## FAUT QUAND MEME GENERER LES DONNEES DE MANIERE LOGIQUES ##########
ploted = []
for i in range(24):
    print(f"Hour {i} :\t{data[data.index.hour == i]['paid_rate'].mean():.2f}")
    print(f"Hour {i} :\t{data[data.index.hour == i]['total_transaction'].mean():.2f}")
    ploted.append(data[data.index.hour == i]['total_transaction'].mean())

# plt.figure(figsize=(20, 10))
# plt.bar(np.arange(24), ploted)
# plt.xticks(np.arange(24))
# plt.show()

morning_data = data[(data.index.hour >= 1) & (data.index.hour <= 6)]

morning_mean_rate = morning_data['paid_rate'].mean()
morning_std_rate = morning_data['paid_rate'].std()
morning_mean_transaction = morning_data['total_transaction'].mean()
morning_std_transaction = morning_data['total_transaction'].std()

mean_general_rate = data['paid_rate'].mean()
std_general_rate = data['paid_rate'].std()
mean_general_transaction = data['total_transaction'].mean()
std_general_transaction = data['total_transaction'].std()

idx = data[data['total_transaction'] < 150].index

nb_elem_missing_by_hour = []
for i in range(24):
    nb_elem_missing_by_hour.append(len(idx[idx.hour == i]))

# idx = sorted(idx, key=lambda x: x.hour)

for i in idx:
    if nb_elem_missing_by_hour[i.hour] >= 0.75 * 273: # missing too much values
        if i.hour >= 1 and i.hour <= 6:
            rd_normal_rate = np.random.normal(morning_mean_rate, morning_std_rate/4)
            rd_normal_transaction = np.random.normal(morning_mean_transaction, morning_std_transaction/2)
            print(f"{i} - enter morning - {data.loc[i][['paid_rate', 'total_transaction']]} - new data : {rd_normal_rate} - {rd_normal_transaction}")            
        else:
            rd_normal_rate = np.random.normal(mean_general_rate, std_general_rate/4)
            rd_normal_transaction = np.random.normal(mean_general_transaction, std_general_transaction/2)
            print(f"{i} - enter general - {data.loc[i][['paid_rate', 'total_transaction']]} - new data : {rd_normal_rate} - {rd_normal_transaction}")
    else:
        mean_spec_rate = data[data.index.hour == i.hour]['paid_rate'].mean()
        std_spec_rate = data[data.index.hour == i.hour]['paid_rate'].std()
        mean_spec_transaction = data[data.index.hour == i.hour]['total_transaction'].mean()
        std_spec_transaction = data[data.index.hour == i.hour]['total_transaction'].std()
        
        rd_normal_rate = np.random.normal(mean_spec_rate, std_spec_rate/4)
        rd_normal_transaction = np.random.normal(mean_spec_transaction, std_spec_transaction/2)    
        print(f"{i} - enter specific - {data.loc[i][['paid_rate', 'total_transaction']]} - new data : {rd_normal_rate} - {rd_normal_transaction}")
    
    input()
    data.loc[i, 'paid_rate'] = rd_normal_rate
    data.loc[i, 'total_transaction'] = round(rd_normal_transaction)

# data.drop(columns=['paid_transaction_count'], inplace=True)
# data.to_csv('../../database/data/data_ogone_norm.csv')


print(f"General mean : {data['paid_rate'].mean()}")
print(f"General std : {data['paid_rate'].std()}")
print(f"General mean : {data['total_transaction'].mean()}")
print(f"General std : {data['total_transaction'].std()}")

for i in range(24):
    print(f"Hour {i} :\t{data[data.index.hour == i]['paid_rate'].mean():.2f}")
    print(f"Hour {i} :\t{data[data.index.hour == i]['total_transaction'].mean():.2f}")





# data = data.loc['2023-09-17':'2023-09-23']
# base_data = base_data.loc['2023-09-17':'2023-09-23']

# plt.figure(figsize=(20, 10))
# plt.plot(data.index, data['paid_rate'], label='Generated data')
# plt.plot(base_data.index, base_data['paid_rate'], label='Real data')
# plt.legend()
# plt.show()


exit()








data_norm = data.copy()

# thresh = 150
# data_inf = data[data['total_transaction'] < thresh]
# data_sup = data[data['total_transaction'] >= thresh]


# select the index (timestamp) where the total_transaction is lower than 150
idx = data[data['total_transaction'] < 150].index

# for each hour specify the number of element inside idx
nb_elem_missing_by_hour = []
for i in range(24):
    print(f"Hour {i} : {len(idx[idx.hour == i])}")
    nb_elem_missing_by_hour.append(len(idx[idx.hour == i]))

input()
mean_general_rate = data['paid_rate'].mean()
std_general_rate = data['paid_rate'].std()
mean_general_transaction = data['total_transaction'].mean()
std_general_transaction = data['total_transaction'].std()

cnt = 0

for i in idx:
    if nb_elem_missing_by_hour[i.hour] >= 0.75 * 273:  # if the number of missing elements is greater than 75% of the total number of elements
        rd_normal_rate = np.random.normal(mean_general_rate, std_general_rate/4)
        rd_normal_transaction = np.random.normal(mean_general_transaction, std_general_transaction/2)
    else:
        cnt+=1
        mean_spec_rate = data[data.index.hour == i.hour]['paid_rate'].mean()
        std_spec_rate = data[data.index.hour == i.hour]['paid_rate'].std()
        mean_spec_transaction = data[data.index.hour == i.hour]['total_transaction'].mean()
        std_spec_transaction = data[data.index.hour == i.hour]['total_transaction'].std()
        
        rd_normal_rate = np.random.normal(mean_spec_rate, std_spec_rate/4)
        rd_normal_transaction = np.random.normal(mean_spec_transaction, std_spec_transaction/2)    
    
    data.loc[i, 'paid_rate'] = rd_normal_rate
    data.loc[i, 'total_transaction'] = round(rd_normal_transaction)

print(f"cnt: {cnt}")
print(f"idx: {len(idx)}")

# drop column paid_transaction_count
data.drop(columns=['paid_transaction_count'], inplace=True)
data.to_csv('../../database/data/data_ogone_norm.csv')
exit()


mydata = data.loc['2023-09-17':'2023-09-23']
# mydata = data.loc['2023-09']

# from the 'idx' array, select the indexes that are in september
idx_sept = []
for i in idx:
    if i.month == 9 and i.year == 2023 and i.day >= 17 and i.day <= 23:
        idx_sept.append(i)

# convert the datetime indexes into integers indexes
idx_nb = [mydata.index.get_loc(i) for i in idx_sept]
print(idx_nb)

attribute = 'total_transaction'

print(len(mydata))
print(mydata[attribute].min())
plt.figure(figsize=(20, 10))
plt.plot(np.arange(len(mydata)), mydata[attribute], label='Real data')
# plt.scatter(idx_nb, mydata.loc[idx_sept][attribute], color='r', label='Generated data')

xticks = np.arange(0, len(mydata)+1, 24)
xticks_labels = [f'D{i+1}' for i in range(len(xticks))]
plt.xticks(xticks, xticks_labels)
# plot a vertical line for each day
for xtick in xticks:
    plt.axvline(x=xtick, color='k', linestyle='--', linewidth=0.5)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.autoscale()    
plt.title(f'{attribute} between {mydata.index.min()} and {mydata.index.max()}')
plt.savefig(f'days_normal_{attribute}.png')
plt.show()


exit()



testos = []
for i in range(24):
    print(f"Hour {i}")
    print(f"Mean : {data[data.index.hour == i]['paid_rate'].mean():.2f}")
    print(f"Std : {data[data.index.hour == i]['paid_rate'].std():.2f}")
    print()

    mini = data[data.index.hour == i]['paid_rate'].min()
    maxi = data[data.index.hour == i]['paid_rate'].max()

    testos.append((i, round(data[data.index.hour == i]['paid_rate'].std(), 2), round(data[data.index.hour == i]['paid_rate'].mean(), 2), maxi, mini))

# sort the testos array by the 2th element of each tuple
testos.sort(key=lambda x: x[1])
[print(elem) for elem in testos]

stats(data, data_inf, data_sup)




print(f"Min timestamp: {data.index.min()}")
print(f"Max timestamp: {data.index.max()}")


# compute the mean of the total_transaction_count for each hour
mean_transaction_count_by_hour_enough_values = []
std_transaction_count_by_hour_enough_values = []
mean_transaction_count_by_hour_low_values = []
std_transaction_count_by_hour_low_values = []

mean_rate_by_hour_enough_values = []
std_rate_by_hour_enough_values = []
mean_rate_by_hour_low_values = []
std_rate_by_hour_low_values = []


transaction_count_by_hour = []
for i in range(24):
    print(f"Hour {i}: {len(data_sup[data_sup.index.hour == i])}")
    transaction_count_by_hour.append(len(data_sup[data_sup.index.hour == i]))

    mean_transaction_count_by_hour_enough_values.append(data_sup[data_sup.index.hour == i]['total_transaction'].mean())
    std_transaction_count_by_hour_enough_values.append(data_sup[data_sup.index.hour == i]['total_transaction'].std())
    mean_transaction_count_by_hour_low_values.append(data_inf[data_inf.index.hour == i]['total_transaction'].mean())
    std_transaction_count_by_hour_low_values.append(data_inf[data_inf.index.hour == i]['total_transaction'].std())
    
    mean_rate_by_hour_enough_values.append(data_sup[data_sup.index.hour == i]['paid_rate'].mean())
    std_rate_by_hour_enough_values.append(data_sup[data_sup.index.hour == i]['paid_rate'].std())
    mean_rate_by_hour_low_values.append(data_inf[data_inf.index.hour == i]['paid_rate'].mean())
    std_rate_by_hour_low_values.append(data_inf[data_inf.index.hour == i]['paid_rate'].std())





# keep the index of the hours that have less than 15% elements of the maximum
idx_to_generate = []
for i in range(24):
    if transaction_count_by_hour[i] < 0.15 * max(transaction_count_by_hour):
        idx_to_generate.append(i)

print(f"idx_to_generate: {idx_to_generate}")

max_transaction = max(transaction_count_by_hour)
print(f"Max elem : {max_transaction}")

# for the hours that are in idx_to_generate, generate the data using the mean and std
for i in idx_to_generate:
    
    indexes = data_sup[data_sup.index.hour == i].index # keep only the indexes that are correct

    time_range = []
    min_time = datetime.strptime(str(data[data.index.hour == i].index.min()), '%Y-%m-%d %H:%M:%S')
    max_time = datetime.strptime(str(data[data.index.hour == i].index.max()), '%Y-%m-%d %H:%M:%S')
    # create a time for every day between min_time and max_time, set the hour to i
    while min_time <= max_time:
        time_range.append(min_time)
        min_time += timedelta(days=1)

    for t in time_range:
        if t not in indexes:
            mydata = data.loc[t]
            # check if the data is not far from the mean
            if abs(mydata['total_transaction'] - mean_transaction_count_by_hour_enough_values[i]) > 2 * std_transaction_count_by_hour_enough_values[i]:
                data_norm.loc[t, 'total_transaction'] = round(np.random.normal(mean_transaction_count_by_hour_low_values[i], std_transaction_count_by_hour_low_values[i])/4)
            if abs(mydata['paid_rate'] - mean_rate_by_hour_enough_values[i]) > 2 * std_rate_by_hour_enough_values[i]:
                data_norm.loc[t, 'paid_rate'] = np.random.normal(mean_rate_by_hour_low_values[i], std_rate_by_hour_low_values[i]/4)

    data_norm['diff_transaction'] = data['total_transaction'] - data_norm['total_transaction']
    data_norm['diff_rate'] = data['paid_rate'] - data_norm['paid_rate']


print(data_norm[data_norm['diff_rate'] != 0])


# data_sept = data.loc['2023-09']
# data_norm_sept = data_norm.loc['2023-09']
data_sept = data
data_norm_sept = data_norm

# plot the rate for each hour
plt.figure(figsize=(20, 10))
plt.plot(data_sept.index, data_sept['paid_rate'], label='Real data')
plt.plot(data_norm_sept.index, data_norm_sept['paid_rate'], label='Generated data', linestyle='--')
# display point where the data has been generated
plt.scatter(data_norm_sept[data_norm_sept['diff_rate'] != 0].index, data_norm_sept[data_norm_sept['diff_rate'] != 0]['paid_rate'], label='Generated data', color='r')
plt.xlabel('Timestamp')
plt.ylabel('Paid rate')
plt.title('Paid rate for september')
plt.legend()
plt.show()

exit()




## Ancienne analyse ##

# data_fn = '../database/data/data_ogone.csv'
# fd_name = 'plots/ogone/year'
# data_fn = '../database/data/month/data_ogone.csv'
# fd_name = 'plots/ogone/month'
# # data_fn = '../database/data/month/data_adyen.csv'
# # fd_name = 'plots/adyen/month'

# stats(data_fn, fd_name)
# exit()




# data_fn = '../data/real_data_ogone.csv'

# data = pd.read_csv(data_fn)
# data['timestamp'] = pd.to_datetime(data['timestamp'])
# data.set_index('timestamp', inplace=True)

days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30]

# for threshold in [50, 55, 60, 65, 70]:
    # anomalies = pd.DataFrame(columns=['timestamp', 'paid_rate', 'total_transaction_count'])
    # anomalies.set_index('timestamp', inplace=True)

    # for month in range(1, 10): # only from january to september included
    #     for day in range(1, days_in_months[month-1]+1):
    #         for i in range(24):
    #             hour_data = data.loc[f'2023-{month}-{day} {i}:00']
    #             if hour_data['paid_rate'] < threshold:
    #                 # save the anomaly in the dataframe with the timestamp as index
    #                 anomalies.loc[f'2023-{month}-{day} {i}:00'] = [hour_data['paid_rate'], hour_data['total_transaction_count']]

    # print(f"Number of anomalies: {len(anomalies)}")
    # anomalies.to_csv(f'anomalies_threshold_{threshold}.csv')

    # print(f"\nThreshold: {threshold}")
    # anomalies_data = pd.read_csv(f'anomalies_threshold_{threshold}.csv')
    # anomalies_data['timestamp'] = pd.to_datetime(anomalies_data['timestamp'])
    # anomalies_data.set_index('timestamp', inplace=True)
    # print(f"Number of anomalies: {len(anomalies_data)}")
    
    # print(f"Number of anomalies with less than 100 transactions: {len(anomalies_data[anomalies_data['total_transaction_count'] < 100])}")
    # print(f"Number of anomalies with 100 to 300 transactions: {len(anomalies_data[(anomalies_data['total_transaction_count'] >= 100) & (anomalies_data['total_transaction_count'] < 300)])}")
    # print(f"Number of anomalies with 300 to 1000 transactions: {len(anomalies_data[(anomalies_data['total_transaction_count'] >= 300) & (anomalies_data['total_transaction_count'] < 1000)])}")
    # print(f"Number of anomalies with more than 1000 transactions: {len(anomalies_data[anomalies_data['total_transaction_count'] >= 1000])}")

    
    # # for every hour of the day compute the mean and min/max of the total_transaction_count. And the number of anomalies
    # anomalies_data['index'] = anomalies_data.index
    # anomalies_data['index'] = anomalies_data['index'].dt.hour
    # anomalies_data = anomalies_data.groupby('index').agg({'total_transaction_count': ['mean', 'min', 'max'], 'paid_rate': 'count'})
    # anomalies_data.columns = ['mean', 'min', 'max', 'count']
    # anomalies_data = anomalies_data.reset_index()
    # print(anomalies_data)
    # input()


    # # plot in a bar chart the number of anomalies for each hour
    # anomalies_data['index'] = anomalies_data.index
    # anomalies_data['index'] = anomalies_data['index'].dt.hour
    # anomalies_data = anomalies_data.groupby('index').count()
    # anomalies_data = anomalies_data.rename(columns={'paid_rate': 'count'})
    # anomalies_data = anomalies_data.reset_index()
    # plt.figure(figsize=(20, 10))
    # plt.bar(anomalies_data['index'], anomalies_data['count'])
    # plt.xticks(anomalies_data['index'])
    # # set on the y axis the number of anomalies, use the index to set correctly the value
    # for i in range(len(anomalies_data)):
    #     plt.text(anomalies_data['index'][i], anomalies_data['count'][i], str(anomalies_data['count'][i]), ha='center', va='bottom')
    

    # plt.xlabel('Hour')
    # plt.ylabel('Number of anomalies')
    # plt.title(f'Number of anomalies for each hour with a threshold of {threshold}')

    # plt.savefig(f'anomalies_threshold_{threshold}.png')
    # plt.show()

    
    # # plot in a bar chart the pourcentage of anomalies for each hour (number of anomalies for each hour / total number of anomalies)
    # anomalies_data['index'] = anomalies_data.index
    # anomalies_data['index'] = anomalies_data['index'].dt.hour
    # anomalies_count = []
    # for i in range(24):
    #     anomalies_count.append(len(anomalies_data[anomalies_data['index'] == i]))
    # total_anomalies = sum(anomalies_count)
    # anomalies_percent = [anomalies_count[i] / total_anomalies * 100 for i in range(24)]
    # plt.figure(figsize=(20, 10))
    # plt.bar(np.arange(24), anomalies_percent)
    # plt.xticks(np.arange(24))
    # plt.xlabel('Hour')
    # plt.ylabel('Pourcentage of anomalies')
    # plt.title(f'Pourcentage of anomalies for each hour with a threshold of {threshold}')
    # # # plt.savefig(f'anomalies_threshold_{threshold}_percent.png')
    # plt.show()

# threshold = 60
# anomalies_data = pd.read_csv(f'anomalies_threshold_{threshold}.csv')
# anomalies_data['timestamp'] = pd.to_datetime(anomalies_data['timestamp'])
# anomalies_data.set_index('timestamp', inplace=True)
# print(f"Number of anomalies: {len(anomalies_data)} in total")
# anomalies_data['index'] = anomalies_data.index
# anomalies_data['index'] = anomalies_data['index'].dt.hour

# # for every anomalies during the day > 6 hour, print the paid_rate and the total_transaction_count
# anomalies_data = anomalies_data[anomalies_data['index'] > 6]
# print(f"Number of anomalies: {len(anomalies_data)} after 6 AM")
# print(anomalies_data[['paid_rate', 'total_transaction_count']])

def anomalies_searching(data):
    threshold = 61
    anomalies = pd.DataFrame(columns=['paid_rate', 'total_transaction_count'])
    anomalies.index = pd.to_datetime([])

    for month in range(1, 10): # only from january to september included
        for day in range(1, days_in_months[month-1]+1):
            for i in range(24):
                hour_data = data.loc[f'2023-{month:02d}-{day:02d} {i:02d}:00']
                if hour_data['paid_rate'] < threshold:
                    # save the anomaly in the dataframe with the timestamp as index
                    anomalies.loc[f'2023-{month:02d}-{day:02d} {i:02d}:00'] = [hour_data['paid_rate'], hour_data['total_transaction_count']]


    # for every anomalies after 6AM, print the paid_rate and the total_transaction_count
    anomalies.index = pd.to_datetime(anomalies.index)
    anomalies['hour'] = anomalies.index.hour
    real_anomalies = anomalies[anomalies['hour'] > 6]
    # real_anomalies = anomalies[anomalies['hour'] >= 0]
    # real_anomalies = anomalies[anomalies['total_transaction_count'] > 300]
    print(f"Number of anomalies: {len(real_anomalies)} after 6 AM")
    print(real_anomalies[['paid_rate', 'total_transaction_count']])


    # create a new column in the data dataframe that's called 'truth', it should be false be default and true only for the anomalies contains in real_anomalies
    data['truth'] = False
    for index in real_anomalies.index:
        data.loc[index, 'truth'] = True

    # plot the paid_rate and the total_transaction_count in a scatter plot, color the anomalies in red
    plt.figure(figsize=(20, 10))
    plt.scatter(data['paid_rate'], data['total_transaction_count'], c=data['truth'].apply(lambda x: 'r' if x else 'b'))
    plt.xlabel('Paid rate')
    plt.ylabel('Total transaction count')
    plt.title(f'Paid rate vs Total transaction count with a threshold of {threshold}')
    plt.savefig(f'anomalies_threshold_{threshold}_scatter.png')
    plt.show()

    # save the data dataframe in a csv file named 'real_data_ogone_truth.csv'
    # data.to_csv('../data/real_data_ogone_truth.csv')


###### old data (wrong percentage) ######
# data = pd.read_csv('../../database/data/high_level_precise.csv')
# data['timestamp'] = pd.to_datetime(data['timestamp'])
# data.set_index('timestamp', inplace=True)

# # mettre payment_method en majuscule
# data['payment_method'] = data['payment_method'].str.upper()

# # erreur dans les données, il faut checker si la méthode de paiement est VISA,MASTERCARD,AMEX et que le type de paiement est DATRS_RED alors il faut changer DATRS_RED en DATRS_HID, pareil pour ADYEN_JSHD et ADYEN_RED
# data.loc[(data['payment_method'] == 'VISA') & (data['payment_type'] == 'DATRS_RED'), 'payment_type'] = 'DATRS_HID'
# data.loc[(data['payment_method'] == 'MASTERCARD') & (data['payment_type'] == 'DATRS_RED'), 'payment_type'] = 'DATRS_HID'
# data.loc[(data['payment_method'] == 'AMEX') & (data['payment_type'] == 'DATRS_RED'), 'payment_type'] = 'DATRS_HID'
# data.loc[(data['payment_method'] == 'VISA') & (data['payment_type'] == 'ADYEN_RED'), 'payment_type'] = 'ADYEN_JSHD'
# data.loc[(data['payment_method'] == 'MASTERCARD') & (data['payment_type'] == 'ADYEN_RED'), 'payment_type'] = 'ADYEN_JSHD'
# data.loc[(data['payment_method'] == 'AMEX') & (data['payment_type'] == 'ADYEN_RED'), 'payment_type'] = 'ADYEN_JSHD'
####### END OF OLD DATA #######


data = pd.read_csv('../../database/data/HIGH_LEVEL_PRECISE_202311171105_alldata_highlevel_precise_percent_right_PAID.csv')

data['DATETIME'] = pd.to_datetime(data['DATETIME'])
data.set_index('DATETIME', inplace=True)
data['PAYMENT_METHOD'] = data['PAYMENT_METHOD'].str.upper()

# drop columns 'STATE'
data = data.drop(columns=['STATE'])

colors = ['#c39bd3', '#5dade2', '#76d7c4', '#f7dc6f', '#ec7063']




def plot_paidrate_specific_payement_type():
    data_ogone_hid = data[data['PAYMENT_TYPE'] == 'OGONE_HID']
    # get only the september 2023 month
    data_ogone_hid = data_ogone_hid.loc['2023-09-01 00:00:00':'2023-09-30 23:59:59']

    data_ogone_hid_visa = data_ogone_hid[data_ogone_hid['PAYMENT_METHOD'] == 'VISA'].copy()
    data_ogone_hid_mastercard = data_ogone_hid[data_ogone_hid['PAYMENT_METHOD'] == 'MASTERCARD'].copy()
    data_ogone_hid_amex = data_ogone_hid[data_ogone_hid['PAYMENT_METHOD'] == 'AMEX'].copy()

    # get the index of visa and amex, then get the difference between the two
    visa_index = data_ogone_hid_visa.index
    amex_index = data_ogone_hid_amex.index
    diff = visa_index.difference(amex_index)
    # for each index in the difference, add a row in the amex dataframe with 0 for paid_rate and transaction_count
    for index in diff:
        data_ogone_hid_amex.loc[index] = ['OGONE_HID', 'AMEX', 0.0, 0]

    # print(len(data_ogone_hid_visa), len(data_ogone_hid_mastercard), len(data_ogone_hid_amex))
    # print(data_ogone_hid_visa['REAL_PERCENTAGE'], len(data_ogone_hid_visa))

    attribute = 'REAL_TRANSACTION_COUNT' #'REAL_PERCENTAGE'#
    name_attribute = 'Transaction count' #'Paid rate' #

    fig, ax = plt.subplots(3, 1, figsize=(20, 10))
    # fig.suptitle(f'Paid rate for each hour in september 2023 for OGONE_HID', fontsize=16)
    fig.suptitle(f'{name_attribute} for each hour in september 2023 for OGONE_HID', fontsize=16)
    ax[0].plot(np.arange(len(data_ogone_hid_visa)), data_ogone_hid_visa[attribute], label='VISA', color='b') #marker='o', linestyle='-', markersize=3)
    ax[0].axhline(y=data_ogone_hid_visa[attribute].mean(), color='r', linestyle='--')
    ax[1].plot(np.arange(len(data_ogone_hid_mastercard)), data_ogone_hid_mastercard[attribute], label='MASTERCARD', color='orange') #marker='o', linestyle='-', markersize=3)
    ax[1].axhline(y=data_ogone_hid_mastercard[attribute].mean(), color='r', linestyle='--',)
    ax[2].plot(np.arange(len(data_ogone_hid_amex)), data_ogone_hid_amex[attribute], label='AMEX', color='g') #marker='o', linestyle='-', markersize=3)
    ax[2].axhline(y=data_ogone_hid_amex[attribute].mean(), color='r', linestyle='--',)

    # get the max of each values
    max_global = data_ogone_hid[attribute].max()

    # set the y axis limit to the max of each values
    ax[0].set_ylim(0, max_global + (0.1 * max_global))
    ax[1].set_ylim(0, max_global + (0.1 * max_global))
    ax[2].set_ylim(0, max_global + (0.1 * max_global))

    # display the x axis a point every 12 hours

    # plot vertical lines for each day
    xticks_pos = np.arange(0, len(data_ogone_hid_visa)+1, 24)
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

    ax[0].set_ylabel(name_attribute)
    ax[0].set_xlabel('Hour')
    ax[1].set_ylabel(name_attribute)
    ax[1].set_xlabel('Hour')
    ax[2].set_ylabel(name_attribute)
    ax[2].set_xlabel('Hour')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.autoscale()
    plt.savefig(f'ogone_hid_{attribute}.png')
    plt.show()

    
    # # input()
    # plt.figure(figsize=(20, 10))
    # plt.plot(np.arange(len(data_ogone_hid_visa)), data_ogone_hid_visa['REAL_PERCENTAGE'], label='VISA')
    # plt.plot(np.arange(len(data_ogone_hid_mastercard)), data_ogone_hid_mastercard['REAL_PERCENTAGE'], label='MASTERCARD')
    # plt.plot(np.arange(len(data_ogone_hid_amex)), data_ogone_hid_amex['REAL_PERCENTAGE'], label='AMEX')
    # plt.xlabel('Hour')
    # plt.ylabel('Paid rate')
    # plt.title(f'Paid rate for each hour in september 2023 for OGONE_HID')
    # # plot a vertical line for each day
    # xticks_pos = np.arange(0, len(data_ogone_hid_visa)+1, 24)
    # xticks_labels = [f'D{i+1}' for i in range(len(xticks_pos))]
    # plt.xticks(xticks_pos, xticks_labels)
    # for xtick in xticks_pos:
    #     plt.axvline(x=xtick, color='k', linestyle='--', linewidth=0.5)

    # # plot horizontal lines for the mean of each payment method
    # plt.axhline(y=data_ogone_hid_visa['REAL_PERCENTAGE'].mean(), color='r', linestyle='--',)
    # plt.axhline(y=data_ogone_hid_mastercard['REAL_PERCENTAGE'].mean(), color='r', linestyle='--',)
    # plt.axhline(y=data_ogone_hid_amex['REAL_PERCENTAGE'].mean(), color='r', linestyle='--',)

    # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # plt.autoscale()
    # plt.legend()
    # plt.savefig(f'ogone_hid_paid_rate.png')
    # plt.show()


# select distinct the payment_type
# print(data['payment_type'].unique(), len(data['payment_type'].unique()))
# input()


# plot the paid_rate during the 12.07.2023 00:00:00 and the 14.07.2023 00:00:00
# data = data.loc['2023-07-12 00:00:00':'2023-07-14 23:59:59']
# print(data.index)
# exit()

# # foreach hour
# for i in range(24):
#     print(f"\nHour {i:02d}")
#     hour_data = data.loc[f'2023-07-13 {i:02d}:00']
#     # group by payment_type and sum the paid_rate
#     hour_data = hour_data.groupby('payment_type').sum('paid_rate')
#     print(hour_data)


print(f"Nb unique payment methods : {len(data['PAYMENT_METHOD'].unique())}")
print(data['PAYMENT_METHOD'].unique())
print(f"Nb unique payment types : {len(data['PAYMENT_TYPE'].unique())}")
print(data['PAYMENT_TYPE'].unique())

# for payment_type in data['payment_type'].unique():
#     my_data = data[data['payment_type'] == payment_type]
#     print(f"{payment_type}: {my_data['payment_method'].unique()}")

# my_data = data[data['payment_method'] == 'KLARNA']
# print(my_data['payment_type'].unique())
# input()

def barplot_by_week():
    # on prend 2 semaines, 1 avant et 1 après l'incident
    incident_day = 19 #13
    start = incident_day - 6
    end = incident_day + 6 + 1

    month = 9 #7

    stats_by_day = []
    for i in range(start, end):
        day = f'2023-{month:02d}-{i:02d}'
        data_day = data.loc[day] # on récupère toute les données du jour
        # data_day = data_day[data_day.index.hour > 6] # on enlève les données entre 1h et 6h
        grouped = data_day.groupby('payment_type')
        group_stats = grouped['paid_rate'].mean()

        for payment_type in data['payment_type'].unique():
            if payment_type not in group_stats:
                group_stats.loc[payment_type] = 0.0
        # print(group_stats)
        # input()
        if i == 13:
            print(group_stats)

        group_stats = group_stats.sort_index()
        stats_by_day.append(group_stats)

    # colors = ['#c39bd3', '#5dade2', '#76d7c4', '#82e0aa', '#f7dc6f', '#e59866', 'r']
    colors = ['#c39bd3', '#5dade2', '#76d7c4', '#f7dc6f', '#ec7063']

    p_start = 7
    bar_width = 0.1
    plt.figure(figsize=(20, 10))
    for i, stat in enumerate(stats_by_day[:6]):
        plt.bar(np.arange(len(stat)) + i*bar_width, stat, width=bar_width, label=f'{p_start+i}/07/2023', color=colors[i])
    plt.bar(np.arange(len(stat)) + 6*bar_width, stats_by_day[6], width=bar_width, label=f'13/07/2023', color=colors[6])
    plt.xticks(np.arange(len(stat)), stat.index, rotation=90)
    plt.xlabel('Payment type')
    plt.ylabel('Paid rate mean')
    plt.title(f'Paid rate mean for each payment type for the week of the 7th of july 2023')
    plt.legend()
    # plt.savefig(f'./Week7thJulyAnalysis.png')
    plt.show()

    p_start = 14
    bar_width = 0.1
    plt.figure(figsize=(20, 10))
    for i, stat in enumerate(stats_by_day[7:]):
        plt.bar(np.arange(len(stat)) + i*bar_width, stat, width=bar_width, label=f'{p_start+i}/07/2023', color=colors[i])
    plt.bar(np.arange(len(stat)) + 6*bar_width, stats_by_day[6], width=bar_width, label=f'13/07/2023', color=colors[6])
    plt.xticks(np.arange(len(stat)), stat.index, rotation=90)
    plt.xlabel('Payment type')
    plt.ylabel('Paid rate mean')
    plt.title(f'Paid rate mean for each payment type for the week of the 13th of july 2023')
    plt.legend()
    # plt.savefig(f'./Week13thJulyAnalysis.png')
    plt.show()


def get_data_hour(year, month, day, hour):
    day_hour = f'{year}-{month:02d}-{day:02d} {hour:02d}:00:00'
    data_hour = data.loc[day_hour]
    grouped = data_hour.groupby('PAYMENT_TYPE').sum('REAL_PERCENTAGE')
    # print(grouped)
    for payment_type in data['PAYMENT_TYPE'].unique():
        if payment_type not in grouped.index:
            grouped.loc[payment_type] = 0.0
    grouped = grouped.sort_index()
    
    
    
    return grouped

def barplot_days_before(data_incident, year, month, day, hour, attribute='REAL_PERCENTAGE', scaling=False, save=False, display=True):
    beforeNafter = 4
    data_before = []
    data_after = []
    name_fig = f'./incidentAnalysisByDays_{attribute}_{day:02d}-{month:02d}-{year}_{hour}h.png'
    
    for i in range(beforeNafter, 0, -1):
        data_before.append(get_data_hour(year, month, day-i, hour)[attribute])
    for i in range(1, beforeNafter+1):
        data_after.append(get_data_hour(year, month, day+i, hour)[attribute])

    # plot in a bar chart
    bar_width = 0.1
    plt.figure(figsize=(20, 10))
    for i, data in enumerate(data_before):
        plt.bar(np.arange(len(data_incident)) - (beforeNafter-i)*bar_width, data, width=bar_width, label=f'{beforeNafter-i} days before', color=colors[i])

    plt.bar(np.arange(len(data_incident)), data_incident, width=bar_width, label=f'incident', color='r')

    for i, data in enumerate(data_after):
        plt.bar(np.arange(len(data_incident)) + (i+1)*bar_width, data, width=bar_width, label=f'{i+1} days after', color=colors[beforeNafter-i-1])
        
    plt.xticks(np.arange(len(data_incident)), data_incident.index, rotation=90)
    plt.xlabel('Payment type')
    plt.ylabel(attribute)
    plt.title(f'{attribute} for each payment type for the incident of {day:02d}/{month:02d}/{year} at {hour:02d}h')
    plt.legend()
    if scaling:
        plt.yscale('log')
        plt.title(f'{attribute} for each payment type for the incident of {day:02d}/{month:02d}/{year} at {hour:02d}h (scaled using log)')
        name_fig = f'./incidentAnalysisByDays_SCALED_{day:02d}-{month:02d}-{year}_{hour}h.png'    
        max_value = max(data_incident.max(), max([data.max() for data in data_before]), max([data.max() for data in data_after]))
        plt.text(0, max_value, f'Max value: {max_value:.2f}', ha='left', va='bottom')


    if save: plt.savefig(name_fig)
    if display:
        plt.show()
    else:
        plt.close()

# barplot_days_before()
# exit()

def barplot_hours_before(data_incident, year, month, day, hour, attribute='REAL_PERCENTAGE', scaling=False, save=False, display=True):
    beforeNafter = 4
    data_before = []
    data_after = []
    name_fig = f'./incidentAnalysisByHours_{attribute}_{day:02d}-{month:02d}-{year}_{hour}h.png'

    for i in range(beforeNafter, 0, -1):
        data_before.append(get_data_hour(year, month, day, hour-i)[attribute])
    for i in range(1, beforeNafter+1):
        data_after.append(get_data_hour(year, month, day, hour+i)[attribute])

    # search the max in the data and the column REAL_PERCENTAGE
    # max_value = max(data_incident.max(), max([data.max() for data in data_before]), max([data.max() for data in data_after]))
    input()

    bar_width = 0.1
    plt.figure(figsize=(20, 10))
    for i, data in enumerate(data_before):
        plt.bar(np.arange(len(data_incident)) - (beforeNafter-i)*bar_width, data, width=bar_width, label=f'{beforeNafter-i} hours before', color=colors[i])
    plt.bar(np.arange(len(data_incident)), data_incident, width=bar_width, label=f'incident', color='r')
    for i, data in enumerate(data_after):
        plt.bar(np.arange(len(data_incident)) + (i+1)*bar_width, data, width=bar_width, label=f'{i+1} hours after', color=colors[beforeNafter-i-1])
    plt.xticks(np.arange(len(data_incident)), data_incident.index, rotation=90)
    plt.xlabel('Payment type')
    plt.ylabel(attribute)
    plt.title(f'{attribute} for each payment type for the incident of {day:02d}/{month:02d}/{year} at {hour:02d}h')
    plt.legend()
    if scaling:
        plt.yscale('log')
        plt.title(f'{attribute} for each payment type for the incident of {day:02d}/{month:02d}/{year} at {hour:02d}h (scaled using log)')
        name_fig = f'./incidentAnalysisByHours_SCALED_{attribute}_{day:02d}-{month:02d}-{year}_{hour}h.png'
        max_value = max(data_incident.max(), max([data.max() for data in data_before]), max([data.max() for data in data_after]))
        plt.text(0, max_value, f'Max value: {max_value:.2f}', ha='left', va='bottom')

    if save: plt.savefig(name_fig)
    if display:
        plt.show()
    else:
        plt.close()

def linechart_specific_day():
    # plot the data from the 14 september 2023
    data_ogone_hid = data[data['PAYMENT_TYPE'] == 'OGONE_HID']
    # get only the september 2023 month
    data_ogone_hid = data_ogone_hid.loc['2023-09-19 00:00:00':'2023-09-19 23:59:59']

    data_ogone_hid_visa = data_ogone_hid[data_ogone_hid['PAYMENT_METHOD'] == 'VISA'].copy()
    data_ogone_hid_mastercard = data_ogone_hid[data_ogone_hid['PAYMENT_METHOD'] == 'MASTERCARD'].copy()
    data_ogone_hid_amex = data_ogone_hid[data_ogone_hid['PAYMENT_METHOD'] == 'AMEX'].copy()

    # amex has only 702 values compared to the other that have 720, for the missing values at the same date and hour, we add a entry in the amex containing only 0
    # get the index of visa and amex, then get the difference between the two
    visa_index = data_ogone_hid_visa.index
    amex_index = data_ogone_hid_amex.index
    diff = visa_index.difference(amex_index)
    # for each index in the difference, add a row in the amex dataframe with 0 for paid_rate and transaction_count
    for index in diff:
        data_ogone_hid_amex.loc[index] = ['OGONE_HID', 'AMEX', 0.0, 0]


    print(len(data_ogone_hid_visa), len(data_ogone_hid_mastercard), len(data_ogone_hid_amex))

    # make 3 subplots one for each payment method and plot the paid_rate for each hour
    fig, ax = plt.subplots(3, 1, figsize=(20, 10))
    fig.suptitle(f'Paid rate for each hour in september 2023 for OGONE_HID', fontsize=16)
    ax[0].plot(np.arange(len(data_ogone_hid_visa)), data_ogone_hid_visa['REAL_PERCENTAGE'], label='VISA')
    ax[1].plot(np.arange(len(data_ogone_hid_mastercard)), data_ogone_hid_mastercard['REAL_PERCENTAGE'], label='MASTERCARD')
    ax[2].plot(np.arange(len(data_ogone_hid_amex)), data_ogone_hid_amex['REAL_PERCENTAGE'], label='AMEX')
    ax[0].axhline(y=data_ogone_hid_visa['REAL_PERCENTAGE'].mean(), color='r', linestyle='--',)
    ax[1].axhline(y=data_ogone_hid_mastercard['REAL_PERCENTAGE'].mean(), color='r', linestyle='--',)
    ax[2].axhline(y=data_ogone_hid_amex['REAL_PERCENTAGE'].mean(), color='r', linestyle='--',)
    plt.xlabel('Hour')
    plt.ylabel('Paid rate')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.autoscale()
    plt.legend()
    plt.savefig('specific_day_september_19.png')
    plt.show()


    # plt.figure(figsize=(20, 10))
    # plt.plot(np.arange(len(data_ogone_hid_visa)), data_ogone_hid_visa['REAL_PERCENTAGE'], marker='o', linestyle='-', color='b', label='VISA')
    # plt.plot(np.arange(len(data_ogone_hid_mastercard)), data_ogone_hid_mastercard['REAL_PERCENTAGE'], marker='o', linestyle='-', color='orange', label='MASTERCARD')
    # plt.plot(np.arange(len(data_ogone_hid_amex)), data_ogone_hid_amex['REAL_PERCENTAGE'], marker='o', linestyle='-', color='g', label='AMEX')
    # plt.xlabel('Hour')
    # plt.ylabel('Paid rate')
    # plt.title(f'Paid rate for each hour in september 2023 for OGONE_HID')

    # plot horizontal lines for the mean of each payment method
    # plt.axhline(y=data_ogone_hid_visa['REAL_PERCENTAGE'].mean(), color='b', linestyle='--',)
    # plt.axhline(y=data_ogone_hid_mastercard['REAL_PERCENTAGE'].mean(), color='orange', linestyle='--',)
    # plt.axhline(y=data_ogone_hid_amex['REAL_PERCENTAGE'].mean(), color='g', linestyle='--',)

    # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # plt.autoscale()
    # plt.legend()
    # plt.savefig('specific_day_september_19.png')
    # plt.show()


# linechart_specific_day()

# plot_paidrate_specific_payement_type()

days_important = [
    [2023, 9, 19, 9], # Ogone
    [2023, 7, 13, 9], # Giropay, klarna (problem 1)
    [2023, 7, 13, 13], # Giropay, klarna (problem 2)
    [2023, 7, 10, 13], # Pour comparer avec le 13
    [2023, 5, 2, 12], # EPC down
    [2022, 10, 3, 13], # Ogone -> VISA
    [2022, 9, 5, 12], # TicketShop + Ogone (problem 1)
    [2022, 9, 5, 14], # TicketShop + Ogone (problem 2)
]

attributes = ['REAL_PERCENTAGE', 'REAL_TRANSACTION_COUNT']

for attribute in attributes:
    if attribute == 'REAL_PERCENTAGE':
        scaled = False
    else:
        scaled = True
    for theday in days_important:
        year, month, day, hour = theday
        data_incident = get_data_hour(year, month, day, hour)[attribute]
        barplot_hours_before(data_incident, year, month, day, hour, attribute=attribute, scaling=scaled, save=True, display=False)
        # if day > 4:
        #     barplot_days_before(data_incident, year, month, day, hour, attribute=attribute, scaling=scaled, save=True, display=False)

exit()

# year = 2023 #2022 #2023
# month = 5 #9 #9 #10 #9 #7
# day = 2 #19 #5 #4 #19 #13
# hour = 12 #12 #13 #9

# print('Get incident')
# data_incident = get_data_hour(year, month, day, hour)['paid_rate']
# barplot_hours_before(data_incident, year, month, day, hour, attribute='total_transaction_count', scaling=True)

# check the representation of each payment methods for each payment types
# for pay_type in data['payment_type'].unique():
#     print(f"\n-----------------------------------------------------\n{pay_type}:")
#     my_data = data[data['payment_type'] == pay_type]
#     # compute for each payment_method the percentage that it represents in the total_transaction_count
#     my_data = my_data.groupby('payment_method').sum('total_transaction_count')
#     total = my_data['total_transaction_count'].sum()
#     my_data['percentage'] = my_data['total_transaction_count'].apply(lambda x: x / total * 100)
#     my_data = my_data.sort_values(by='percentage', ascending=False)
#     my_data = my_data[my_data['percentage'] > 1]
#     print(my_data)
#     print("-----------------------------------------------------\n")



