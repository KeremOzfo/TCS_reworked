from train_funcs import *
import numpy as np
from parameters import *
import torch
import random
import datetime
import os


if __name__ == '__main__':
    args = args_parser()
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    simulation_ID = int(random.uniform(1,9999))
    print('device:',device,'pytorch:',torch.__version__)
    args = args_parser()
    results = []
    x = datetime.datetime.now()
    date = x.strftime('%b') + '-' + str(x.day)
    if args.mode == 'tcs':
        newFile ='TCS-Quantize_{}-LSGD_{}-Dirich_{}-ID-{}'.format(args.quantization,args.LSGDturn,args.alfa,simulation_ID)
    elif args.mode == 'topk':
        newFile = 'TopK-Quantize_{}-Dirich_{}-ID-{}'.format(args.quantization,args.alfa,
                                                                   simulation_ID)
    else:
        raise NotImplementedError('incompatible mode')
    if not os.path.exists(os.getcwd() + '/Results'):
        os.mkdir(os.getcwd() + '/Results')
    n_path = os.path.join(os.getcwd(), 'Results', newFile)
    for i in range(args.num_trail):
        if args.mode == 'tcs':
            accs = train(args, device)
        elif args.mode == 'topk':
            accs = train_topk(args, device)
        if i == 0:
            os.mkdir(n_path)
            f = open(n_path + '/simulation_Details.txt', 'w+')
            f.write('simID = ' + str(simulation_ID) + '\n')
            f.write('############## Args ###############' + '\n')
            for arg in vars(args):
                line = str(arg) + ' : ' + str(getattr(args, arg))
                f.write(line + '\n')
            f.write('############ Results ###############' + '\n')
            f.close()
        s_loc = '{}-Quantize_{}-LSGD_{}-Dirich_{}-acc'.format(args.mode,args.quantization,args.LSGDturn,args.alfa) +str(i)
        s_loc = os.path.join(n_path,s_loc)
        np.save(s_loc,accs)
        f = open(n_path + '/simulation_Details.txt', 'a+')
        f.write('Trial ' + str(i) + ' results at ' + str(accs[-1]) + '\n')
        f.close()