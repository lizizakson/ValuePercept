def Pmaxelements(list1, P): 
    final_list = []
    
    list_temp = list(map(abs, list1)) #convert all elements to absolute value
    
    list_temp.sort(reverse = True) #sort the elements from the highest to the lowest element
    
    num_elements = int(len(list_temp)*P) #num of hiest elements needed to be found according the % given
    
    final_list = list_temp[0:num_elements] #take the highest X elements from the original list
    
    threshold = final_list[num_elements] #the minimum highest element in the list
    
    #Get the indices of these highest elements (in the original list)
    final_list_ind = [] #create new list to store the indices of the highest elements from the original list in
    list1_abs = list(map(abs, list1)) #conveet the original list to abs values (not sorted)
    for elem in final_list:
        index = list1_abs.index(elem)
        if index not in final_list_ind: #if the index does not already exict in the final indices list
            final_list_ind.append(index)
        else:  #if the index exicts in the final indices list
            index = len(list1_abs[:index+1]) + list1_abs[index:].index(elem) #find the next index which this elements appears in
            print(len(list1_abs[:index+1]))
            print(list1_abs[index:])
            print(list1_abs[index:].index(elem))
            #print(list1_abs[index+1:])
            final_list_ind.append(index)  
    
    return final_list, final_list_ind

test_list = [9 ,9 ,2, -9]

test1 = Pmaxelements(test_list, 0.75)
test1