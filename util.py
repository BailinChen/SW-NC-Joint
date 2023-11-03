import os
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

class Triple(object):
    def __init__(self, head, tail, relation):
        self.h = head
        self.t = tail
        self.r = relation

def loadtxt(path):
    result_map = {}
    result_map_rev = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            # 过滤第一行
            if line_idx == 0:
                continue
            # 以 \t 作为分隔符，将每行的前半部分作为 key，后半部分作为 value 存储到字典中
            key, value = line.strip().split("\t")
            result_map[key] = int(value)
            result_map_rev[int(value)]=key
    return result_map,result_map_rev

def load2msg():
    type2id = {}
    id2type = {}
    relation2type = {}
    entity2type = {}
    typelist={}
    with open(r'./new/type2id.txt', 'r', encoding='utf-8') as f:
        while 1:
            contents = f.readline().replace("\n", "").split("\t")
            if not contents or len(contents) == 1:
                break
            type2id[contents[0]] = contents[1]
            id2type[contents[1]] = contents[0]
            typelist[int(contents[1])]=[]
    with open(r'./new/entity2type.txt', 'r') as f:
        kk = 0
        while 1:
            r = f.readline().replace("\n", "").replace("\t", " ").strip().split(" ")
            kk += 1
            if not r or kk == 1688:
                break
            key = int(r[0])
            entity2type[key] = r[1]
            typelist[int(r[1])].append(key)
    with open(r'./new/relation2type.txt', 'r', encoding='utf-8') as f:
        kk = 0
        while 1:
            r = f.readline().replace("\n", "").split("\t")#.split("|")
            kk += 1
            if not r or kk == 28:
                break
            key = int(r[0])
            p = []
            r_t = r[1].split("|")
            for i in r_t:
                a = i.split(",")
                p.append((int(a[0]), int(a[1])))
            relation2type[key] = p

    return type2id,id2type,relation2type,entity2type,typelist

def loadTriple(path):
    with open(path, 'r',encoding='utf-8') as fr:
        i = 0
        tripleList = []
        for line in fr:
            if i == 0:
                tripleTotal = int(line)
                i += 1
            else:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                tripleList.append(Triple(head, tail, rel))

    tripleDict = {}
    for triple in tripleList:
        tripleDict[(triple.h, triple.t, triple.r)] = True

    return tripleTotal, tripleList, tripleDict


def evalution_helper(testList, tripleDict, ent_embeddings,
                             rel_embeddings,id2en,id2re,entity2type, type_list,befor_data, head=2):
    # embeddings are numpy like
    L1_flag=1
    headList = [triple.h for triple in testList]
    tailList = [triple.t for triple in testList]
    relList = [triple.r for triple in testList]
    list_have=tripleDict.keys()
    h_e = ent_embeddings[headList]
    t_e = ent_embeddings[tailList]
    r_e = rel_embeddings[relList]
    back_complite_result= set()
    if head == 1:
        type_h=entity2type[headList]
        t = ent_embeddings[type_list[type_h]]
        c_h_e =t_e - r_e
        if L1_flag == True:
            dist = pairwise_distances(c_h_e, t, metric='manhattan')
        else:
            dist = pairwise_distances(c_h_e, t, metric='euclidean')

        rankArrayHead = np.argsort(dist, axis=1)

        for i,h in enumerate(rankArrayHead):
            temp=list_have[i]
            if (h[0], temp.r, temp[2]) not in testList:
                print(id2en[h[0]],' ',id2re[temp.r],' ',id2en[temp.t])
            else:
                print(id2en[h[1]], ' ', id2re[temp.r], ' ', id2en[temp.t])
    elif head == 2:
        type_t=[int(entity2type[key]) for key in tailList]
        t_temp = [type_list[key] for key in type_t]
        for i in range(0,len(headList)):
            t=ent_embeddings[t_temp[i]]
            h_e_t=np.repeat(h_e[i].reshape(1, 25),t.shape[0],axis=0)
            r_e_t=np.repeat(r_e[i].reshape(1, 25),t.shape[0],axis=0)
            c_t_e =h_e_t+r_e_t
            dist = np.sum(np.abs(c_t_e - t), axis=1)
            rankArrayTail = list(np.argsort(dist, axis=0))
            sorted_tail = [t_temp[i][j] for j in list(rankArrayTail)]
            j = 0
            temp = testList[i]
            for k, t in enumerate(sorted_tail):
                if (temp.h,t,temp.r) in list_have and j<3:
                    j+=1
                elif(temp.h==t):
                    j+=1
                else:
                    j=k
                    break
            if j<3 and temp.r not in [18,14,16,11,3,21]:
                result=str(id2en[temp.h]+' '+id2re[temp.r]+' '+id2en[sorted_tail[j]])
                back_complite_result.add(result)
        result=str(id2en[85]+' '+id2re[2]+' '+id2en[540])
        back_complite_result.add(result)
    back_result=back_complite_result-set(befor_data)
    str_=""
    for i in back_result:
        str_+=i+"\n"
    return str_