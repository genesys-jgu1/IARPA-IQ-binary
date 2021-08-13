import pickle as pkl
import numpy as np
import os
from collections import defaultdict
import glob
    
device_list = ['m1001','m1005','m1007','m1008','m1009','m10010','m10011']
distance_list = ['6ft','9ft','12ft','15ft']

#pred_base = '/home/nasim/MachineLearning/results/uav-indoor-6cnns/'
pred_base = '/mnt/nas/nasim/UAV/results/uav-chop10-12cnns/'
#pred_base = '/home/nasim/MachineLearning/results/uav-indoor-3cnns/'
#pred_base = '/home/nasim/MachineLearning/results/uav-indoor-3cnn-oldanduseless/'


def pred_file_list_maker(pred_base):
    pred_paths = glob.glob(pred_base+'*/preds.pkl')
    return pred_paths

def pred_reader(pred_file):
    with open (pred_file,'rb') as handle:
        pic_file = pkl.load(handle)
    ex_pred = pic_file['preds_ex']
    slice_pred = pic_file['preds_slice']
    return ex_pred,slice_pred

def preds_former(ex_pred):
    new_dict = defaultdict()
    for ex in ex_pred:
        # the keys must form a structure like:  uav3_12ft_burst2_14     ??????
        new_key = ex.split('/')[-2]+'_'+ex.split('/')[-1].split('.')[-2]
        new_dict[new_key] = ex_pred[ex]
    return new_dict

def multiburst_acc(ex_pred, multi_degree, sets_in_test_list, slice_pred, multiburst_type):
    """for key in ex_pred:
        print ex_pred[key]"""

    ex_list_in_dict = ex_pred.keys()
    multiburst_preds_dict = defaultdict(list) 
    multiburst_preds_dict_count = defaultdict() 
    for device in device_list:
        for distance in distance_list:
             
            # we are inside set 4
            for each_set in sets_in_test_list:
                this_list = [int(x.split('_')[-1]) for x in ex_list_in_dict if x.split('_')[:3] == [device,distance,each_set]]
                this_list.sort() 
                
                initial_point = 0
                burst_count = len(this_list)/multi_degree
                #print 'burst_ count ' + str(burst_count)
                for _ in range(burst_count):
                    
                    #start creating multi_burst_preds_dict
                    vector = np.zeros(len(device_list))
                    new_key = device+'_'+distance+'_'+each_set 
                    voting_list = []
                    voting_index_list = []
                    count_list = []
                    
                    for i in range(multi_degree):
                        this_key = device+'_'+distance+'_'+each_set+'_'+str(this_list[initial_point])
                        
                        """ check if type of multiburst is prob sum or voting"""
                        if multiburst_type == 'prob_sum':
                            vector += ex_pred[this_key]
                        
                        elif multiburst_type == 'voting':   #sub-ex voting
                            this_predicted_index = np.argmax(ex_pred[this_key])
                            #print slice_pred[this_key][1]/slice_pred[this_key][0]
                            this_scaled_prediction = (1.0*slice_pred[this_key][1]/slice_pred[this_key][0])*ex_pred[this_key][this_predicted_index]
                            count_list.append((slice_pred[this_key][0],slice_pred[this_key][1]))
                            voting_list.append(this_scaled_prediction)
                            voting_index_list.append(this_predicted_index)
                        
                        new_key += '_'+str(this_list[initial_point])
                        initial_point += 1
                    
                    # when we get here new key is ready
                    # for prob_sum mode, the vector is also ready
                    # for each burst we update and save the vector in voting mode

                    """voting ensemble begins"""
                    if multiburst_type == 'voting':

                        # if you want to combine mini-examples in first layer:
                        # among 10 mini_ex:
                        winning_mini_ex_count = count_list[voting_list.index(max(voting_list))]
                        #print new_key
                        vector = np.zeros(len(device_list))
                        winning_class = voting_index_list[voting_list.index(max(voting_list))]

                        #suppressing all the others and only keeping the winning class
                        vector[winning_class] = max(voting_list)
                    """voting ensemble ends"""
                    
                    multiburst_preds_dict[new_key] = vector
                    multiburst_preds_dict_count[new_key] = winning_mini_ex_count
                    #initial_point -= 2
    return multiburst_preds_dict, multiburst_preds_dict_count

def argmax_acc_calculator(ex_pred):
    total_count = 0
    correct_count = 0
    for key in ex_pred:

        true_index = int(device_list.index(key.split('_')[0]))
        predicted_index = np.argmax(ex_pred[key])
        if true_index == predicted_index:
            #print key
            correct_count += 1
        #else:
            #print 'wrong ex ' +key
        total_count += 1
    return 1.0*correct_count/total_count

def per_distance_acc(ex_pred):
    total_count = np.zeros(len(distance_list))
    correct_count = np.zeros(len(distance_list))
    for key in ex_pred:
        true_index = int(device_list.index(key.split('_')[0]))
        predicted_index = np.argmax(ex_pred[key])
        distance_index = int(distance_list.index(key.split('_')[1]))
        if true_index == predicted_index:
            correct_count[distance_index] += 1
        total_count[distance_index] += 1    
    return 1.0*correct_count/total_count

def per_device_acc(ex_pred):
    total_count = np.zeros(len(device_list))
    correct_count = np.zeros(len(device_list))
    for key in ex_pred:
        true_index = int(device_list.index(key.split('_')[0]))
        predicted_index = np.argmax(ex_pred[key])
        if true_index == predicted_index:
            correct_count[true_index] += 1
        total_count[true_index] += 1    
    return 1.0*correct_count/total_count



def ensemble(ex_pred_dicts,slice_pred_dicts):
    overall_ex_dict = {}
    overall_slice_dict = {}
    correct_ex = 0
    total_ex =0
    print len(ex_pred_dicts)
    
    for key in ex_pred_dicts[0]:
        
        true_index = device_list.index(key.split('_')[0])

        """creating vectors and weights"""
        #count = []
        overall_vect = 0
        predicted_index_list = []
        elem = []

        for i in range(len(ex_pred_dicts)):
            this_ex_pred = ex_pred_dicts[i]
            vect = this_ex_pred[key]
            this_slice_pred = slice_pred_dicts[i]

            #count.append(slice_pred_dicts[i][key])   
            weight = (1.0*this_slice_pred[key][1])/this_slice_pred[key][0]
            predicted_index = np.argmax(vect)
            predicted_index_list.append(predicted_index)
            #overall_vect += vect*weight
            elem.append(vect[predicted_index]*weight)

        """ Now list of elem for this key (example) is ready """
        overall_pred = predicted_index_list[elem.index(max(elem))]
            
        """ we make a fake vector with all elements set to zero only the max element is set to value"""
        fake_vector = np.zeros(len(device_list))
        fake_vector[overall_pred] = max(elem)
        
        overall_ex_dict[key] = fake_vector
        overall_slice_dict[key] = (slice_pred_dicts[0][key][0],max(elem))
        
        if overall_pred == true_index:
        #if true_index == predicted_index_1 or true_index==predicted_index_2 or true_index == predicted_index_3:
            correct_ex += 1
        total_ex += 1

    print (1.0*correct_ex)/total_ex
    print correct_ex,total_ex

    total_ex_dev = np.zeros(len(device_list))
    correct_ex_dev = np.zeros(len(device_list))

    """for ex in overall_ex_dict:
        true_index = device_list.index(ex.split('_')[0])
        if true_index == overall_ex_dict[ex]:
            correct_ex_dev[true_index] += 1
        total_ex_dev[true_index] += 1"""
    """for i in range(len(device_list)):
        print (1.0*correct_ex_dev[i])/total_ex_dev[i]"""
    
    return overall_ex_dict, overall_slice_dict

if __name__ == '__main__':
    multiburst_degree = 10 
    sets_in_test_list = ['4']

    pred_list = pred_file_list_maker(pred_base)
    print pred_list

    ex_pred_list_of_dict = []
    slice_pred_list_of_dict = []
    ex_pred_list_of_dict_after_multiburst = []
    slice_pred_list_of_dict_after_multiburst = []

    #pred_list = ['/home/nasim/MachineLearning/results/uav-indoor/uav-indoors-train123-test4/preds.pkl']
    #pred_list = ['/home/nasim/MachineLearning/results/uav-indoor-sliding/preds-testlast10percent.pkl']
    #pred_list = ['/home/nasim/MachineLearning/results/uav-chop10-resnet/old_preds_prob_sum.pkl']
    index = 0
    
    for pred_file in pred_list:
        ex_pred,slice_pred = pred_reader(pred_file)
        ex_new_dict = preds_former(ex_pred)
        slice_new_dict = preds_former(slice_pred)
        ex_pred_list_of_dict.append(ex_new_dict)
        slice_pred_list_of_dict.append(slice_new_dict)
        #multiburst_dict = multiburst_acc(ex_pred_list_of_dict[index],multiburst_degree,sets_in_test_list,slice_pred_list_of_dict[index],'prob_sum')
        
        print '------------------------------------------'
        print 'multiburst degree is: ' +str(multiburst_degree)
        print pred_file
        overall_acc = argmax_acc_calculator(ex_new_dict)
        print 'overall accuracy: ' +str(round(overall_acc, 2)) 
        """multiburst_dict = multiburst_acc(ex_new_dict, multiburst_degree,sets_in_test_list,slice_new_dict, 'prob_sum')
        overall_acc_multi = argmax_acc_calculator(multiburst_dict)
        print 'overall accuracy after multi burst prob_sum:' +str(round(overall_acc_multi, 2))"""
        multiburst_dict, multiburst_count_dict = multiburst_acc(ex_new_dict,multiburst_degree,sets_in_test_list, slice_new_dict, 'voting')
        overall_acc_multi = argmax_acc_calculator(multiburst_dict)
        print 'overall accuracy after multi burst voting:' +str(round(overall_acc_multi, 2))


        # per distance and device acc for different sub-cnns
        """per_dist_acc = per_distance_acc(ex_new_dict)
        per_dist_acc = map(lambda x: round(x,2), per_dist_acc)
        print 'per distance accuracy: '
        print per_dist_acc        
        
        per_dev_acc = per_device_acc(ex_new_dict)
        per_dev_acc = map(lambda x: round(x,2), per_dev_acc)
        print 'per device accuracy: ' 
        print per_dev_acc"""       
        print '------------------------------------------'

        # if you are doing multiburst in the first layer: save them in a list
        ex_pred_list_of_dict_after_multiburst.append(multiburst_dict)
        slice_pred_list_of_dict_after_multiburst.append(multiburst_count_dict)
        index += 1

    # now calculate the ensemble  
    #overall_ex_pred, overall_slice_pred = ensemble(ex_pred_list_of_dict,slice_pred_list_of_dict)

    overall_ex_pred, overall_slice_pred = ensemble(ex_pred_list_of_dict_after_multiburst,slice_pred_list_of_dict_after_multiburst)

    before_multi = argmax_acc_calculator(overall_ex_pred)
    print 'acc before multi after ensemble is: ' +str(before_multi)
    

    with open('./horizontal_vertical_aggregation.pkl','wb') as handle:
        pkl.dump(overall_ex_pred,handle)



    #final_acc = argmax_acc_calculator(multiburst_acc(overall_ex_pred, multiburst_degree, sets_in_test_list, overall_slice_pred, 'voting')[0])
    #print 'overall accuracy after multi burst voting:' +str(round(final_acc, 2))

    

    """per_dist_acc = per_distance_acc(multiburst_acc(overall_ex_pred, multiburst_degree, sets_in_test_list, overall_slice_pred, 'voting')[0])
    per_dist_acc = map(lambda x: round(x,2), per_dist_acc)
    print 'per distance accuracy: '
    print per_dist_acc""" 
