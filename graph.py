import numpy as np
import matplotlib.pyplot as plt
from os import *
import math
from itertools import cycle
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def table(dic,save_csv = False):
    panda = pd.DataFrame(dic)
    print(panda)
    if save_csv:
        panda.to_csv('results.csv')

def load_results(file):
    dic = {'mode':[],'Q4':[],'H':[],'Dirich':[],'Result':[],'std':[]}
    results = []
    for result in listdir(file):
        res_path = path.join(file,result)
        details = result.split('-')
        for key,detail in zip(dic,details):
            info = detail.split('_')
            dic[key].append(info[-1])
            if key=='Dirich':
                break
        res, avg, stdev = compile_results(res_path)
        results.append(res)
        dic['Result'].append(np.round(avg,3))
        dic['std'].append(np.round(stdev,3))
    table(dic,True)
    return dic, results

def compile_results(adress):
    f_results = []
    first = True
    results = None
    for dir in listdir(adress):
        if dir[0] !='s':
            vec = np.load(path.join(adress,dir))
            final_result = vec[len(vec)-1]
            f_results.append(final_result)
            if first:
                results = vec
                first = False
            else:
               results += vec
    avg = np.average(f_results)
    st_dev = np.std(f_results)
    results = np.array(results) / len(f_results)
    return results, avg,st_dev

def legend_maker(dic):
    legends = []
    total = len(dic['mode'])
    for i in range(total):
        mode = dic['mode'][i]
        lsgd = dic['H'][i]
        alfa = dic['Dirich'][i]
        quan = '-Q4' if dic['Q4'][i] == 'True' else ''
        leg = '{}-H={}-\u03B1={}{}'.format(mode,lsgd,alfa,quan)
        legends.append(leg)
    return legends


def graph(data, dic,interval=1):
    marker = ['s', 'v', '+', 'o', '*']
    color = ['r', 'c', 'b', 'g','y','b','k']
    linestyle =['-', '--', '-.', ':']
    linecycler = cycle(linestyle)
    colorcycler = cycle(color)
    markercycler = cycle(marker)
    legends = legend_maker(dic)
    fig, ax = plt.subplots()
    final_accs = []
    for d,legend in zip(data,legends):
        x_axis = []
        l = next(linecycler)
        m = next(markercycler)
        c = next(colorcycler)
        for i in range(0,len(d)):
            x_axis.append(i*interval)
        ax.plot(x_axis,d, marker= m ,linestyle = l ,markersize=2, label=legend)
        final_accs.append(d[-1])
    #plt.axis([0, 30,70 ,90])
    #plt.axis([145,155,88,92])
    #plt.axis([290, 300, 90, 95])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    axins1 = zoomed_inset_axes(ax, zoom=16, loc=10)

    for d, legend in zip(data, legends):
        x_axis = []
        for i in range(0, len(d)):
            x_axis.append(i * interval)
        axins1.plot(x_axis, d)
    axins1.grid(True)
    ax.plot()

    # SPECIFY THE LIMITS
    x1, x2, y1, y2 = 295, 300, min(final_accs)-0.3, max(final_accs) +0.3
    axins1.set_xlim(x1, x2)
    axins1.set_ylim(y1, y2)
    # IF SET TO TRUE, TICKS ALONG
    # THE TWO AXIS WILL BE VISIBLE
    plt.xticks(visible=False)
    plt.yticks(visible=False)

    mark_inset(ax, axins1, loc1=1, loc2=2, fc="none", ec="0.2")
    plt.tight_layout()
    plt.show()



def concateresults(dirsets):
    all_results =[]
    for set in dirsets:
        all_results.append(compile_results(set)[0])
        print(compile_results(set)[1])
    return all_results

def dirichlet_show(alpha,num_worker=10):
    colors = ['b','g','r','c','m','y','slategrey','maroon','indigo','lime']
    dirichlet = np.repeat(alpha, 10)
    dirichlet_vec = np.random.dirichlet(dirichlet, num_worker)
    clas_dist = np.sum(dirichlet_vec, axis=0)
    #print(dirichlet_vec)
    for i,vec in enumerate(dirichlet_vec):
        for cls in range(10):
            vec[cls] *= 1/clas_dist[cls]
    dirichlet_vec*=5000

    for i,vec in enumerate(dirichlet_vec):
        for cls in range(10):
            if cls==0:
                plt.barh(i+1,vec[cls],color=colors[cls])
            else:
                plt.barh(i+1,vec[cls],left=np.sum(vec[0:cls]),color=colors[cls])
    plt.tight_layout()
    plt.show()



dic,results = load_results('Results/topk')
graph(results,dic)
#dirichlet_show(0.5,10)