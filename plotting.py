# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import json
import numpy as np

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)
#%% plot test results
plt.close('all')
root = 'logs/2012012257'

with open(root + '/test_result.json' , 'r') as file:
    test_results = json.load(file)

with open('alg_performance/low.json' , 'r') as file:
    low_per = json.load(file)    

with open('alg_performance/high.json' , 'r') as file:
    high_per = json.load(file) 

TP = np.array(test_results['TP'])
TC = np.array(test_results['TC'])
FP = np.array(test_results['FP'])
PNR = np.array(test_results['PNR'])
sort = np.argsort(PNR)
Counts = np.array(test_results['Counts'])

TP_low = np.array(low_per['TP'])
TC_low = np.array(low_per['TC'])
FP_low = np.array(low_per['FP'])
PNR_low = np.array(low_per['PNR'])

TP_high = np.array(high_per['TP'])
TC_high = np.array(high_per['TC'])
FP_high = np.array(high_per['FP'])
PNR_high = np.array(high_per['PNR'])

idx = Counts[sort] < 300
plt.figure()
plt.plot(PNR[sort][idx] , TP[sort][idx] / TC[sort][idx] , label = 'TPR') 
plt.plot(PNR[sort][idx] , FP[sort][idx] / TC[sort][idx], label = 'FDR')
plt.plot(PNR_low, TP_low / TC_low, '--b', label = 'TPR benchmark')
plt.plot(PNR_low, FP_low / TC_low, '--', color = 'red', label = 'FDR benchmark')
plt.grid('on')
plt.xlabel('Peak to Noise Ratio')
plt.title('Less than 300 counts')
plt.legend()


idx = np.invert(idx)
plt.figure()
plt.plot(PNR[sort][idx] , TP[sort][idx] / TC[sort][idx], label = 'TPR')
plt.plot(PNR[sort][idx] , FP[sort][idx] / TC[sort][idx], label = 'FDR')
plt.plot(PNR_high, TP_high / TC_high, '--b', label = 'TPR benchmark')
plt.plot(PNR_high, FP_high / TC_high, '--', color = 'red', label = 'FDR benchmark')
plt.grid('on')
plt.xlabel('Peak to Noise Ratio')
plt.legend()
plt.title('More than 1000 counts')



# idx = torch.arange(98500,98500 + mini_batch)
# outputs = net(inputs[:,:,idx].float())
# pred = torch.max(outputs, dim = 1)[1].data.numpy()
# plt.figure()
# plt.plot(targets[0,idx])
# plt.plot(pred[0,:])
# plt.figure()
# plt.plot(targets[0,idx])
# plt.plot(10**outputs[0,1,:].data.numpy())
# # plt.plot(outputs[0,0,:].data.numpy())
# #plt.plot(pred[0,:])
# plt.figure()
# plt.plot(inputs[0,0,idx])
# plt.plot(inputs[0,1,idx])
# plt.plot(inputs[0,2,idx])
# plt.plot(inputs[0,3,idx])
# plt.plot(pred[0,:]*torch.max(inputs[:,:,idx]).numpy())



# t = targets.data.numpy()[0,idx]
# p = pred[0,:]

# TP, TC, FP, TC_p = model_performance(t,p,0.3)
# print(TP,TC,FP,TC_p)