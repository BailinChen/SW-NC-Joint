#python 为java提供入口 用于使用模型
import torch
from flask import Flask
from flask import request
import flask, json
from flask_cors import *
import TransE
from model_paper2 import *
import util
import mysql.connector
import ast
'''
flask： web框架，通过flask提供的装饰器@server.route()将普通函数转换为服务
登录接口，需要传url、username、passwd
'''
app = Flask(__name__)

CORS(app,supports_credentials=True)
n_entity=1687
n_relation=27
VECTOR_LENGTH=25
NORM=1
MARGIN=6.0
# transe = TransEModel(n_entity, n_relation, VECTOR_LENGTH, p_norm=NORM, margin=MARGIN)
transe = torch.load("./new/XMKG_MODEL.tar").cpu()
# transe.load_state_dict(checkpoint['model_state_dict'])
en_map,id2en=util.loadtxt('./new/entity2id.txt')
re_map,id2re=util.loadtxt('./new/relation2id.txt')
type2id,id2type,relation2type,entity2type,typelist=util.load2msg()
tripleTotal, tripleList, tripleDict=util.loadTriple('./new/triple2id.txt')
mydb = mysql.connector.connect(
  host="127.0.0.1",
  user="root",
  password="root",
  database="xmgraph1"
)

@app.route('/', methods=['get', 'post'])
def index():
    return "Hello, World!"



#返回三元组得分与三元组正确与否
@app.route('/triple_correct', methods=['get', 'post'])
def triple_correct():
    transe.eval()
    thre = {}
    with open("./new/relation_thr.txt", "r",encoding="utf-8") as f:
        for l in f:
            msg = l.split("\t")
            thre[msg[0]] = [msg[1], msg[2]]
    head = request.values.get('head')
    relation = request.values.get('relation')
    tail = request.values.get('tail')
    #all_msg=request.values.get('allin')
    #判断是否存在实体与关系
    h,r,t=en_map[head],re_map[relation],en_map[tail]
    valid_data_vec = torch.tensor([h,r,t], dtype=torch.int64).view([-1,3])
    dist = float(transe.forward_self(valid_data_vec)[0])
    r_thr=float((thre[str(r)][1]).replace("\n",""))
    #处理与判断
    judje = "知识正确" if dist <= r_thr else "知识错误"
    type_doub=relation2type[r]
    h_t,t_t=entity2type[h],entity2type[t]
    if (int(h_t),int(t_t)) not in type_doub:
        judje="错误，所选实体类型与关系不匹配。"
        dist="TYPE ERROR!"
    #形成最终
    # print( json.dumps({"ms":messg}, ensure_ascii=False))
    return json.dumps({"judje":judje,"dist":dist,"r_thr":r_thr}, ensure_ascii=False)



#返回关系阈值信息
@app.route('/threshold', methods=['get', 'post'])
def threshold():
    thre={}
    with open("./new/relation_thr.txt", "r",encoding="utf-8") as f:
        for l in f:
            msg=l.split("\t")
            thre[msg[0]]=[msg[1],msg[2]]
    return json.dumps({"ms":thre}, ensure_ascii=False)

#返回补全结果
@app.route('/complite', methods=['get', 'post'])
def complite():
    comp_msg={}
    ent_embeddings=transe.ent_embeddings.weight.detach().numpy()
    rel_embeddings = transe.rel_embeddings.weight.detach().numpy()
    mydb.reconnect()
    mycursor = mydb.cursor()
    sql = "SELECT entity_one,SUBSTRING_INDEX(relation_name, '（', 1),entity_two FROM triple_comp"
    mycursor.execute(sql)
    befor_data=[]
    myresult = mycursor.fetchall()
    for x in myresult:
        try:
            befor_data.append((x[0]+' '+x[1]+' '+x[2]))
        except:
            continue
    msg=util.evalution_helper(tripleList, tripleDict, ent_embeddings,rel_embeddings,id2en,id2re,entity2type, typelist,befor_data)
    msg.replace(" ","      ")
    mycursor.close()
    mydb.close()
    return json.dumps({"ms":msg}, ensure_ascii=False)



#host="0.0.0.0" 代表谁都可以访问，可以加nginx处理
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=60015, debug=True)  ###指定端口、host设为0.0.0.0代表不管几个网卡，任何ip都可以访问
