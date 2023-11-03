#python 为java提供入口 用于使用模型
import torch
from flask import Flask
from flask import request
import flask, json
from flask_cors import *
import TransE
import util
import ast
'''
flask： web框架，通过flask提供的装饰器@server.route()将普通函数转换为服务
登录接口，需要传url、username、passwd
'''
app = Flask(__name__)

CORS(app,supports_credentials=True)
n_entity=14951
n_relation=1345
VECTOR_LENGTH=200
NORM=1
MARGIN=3.0
transe = TransE.TransE(n_entity, n_relation, VECTOR_LENGTH, p_norm=NORM, margin=MARGIN)
transe.load_checkpoint("./checkpoint_200_3_TransE.tar")
en_map=util.loadtxt('./entity2id.txt')
re_map=util.loadtxt('./relation2id.txt')

@app.route('/', methods=['get', 'post'])
def index():
    return "Hello, World!"


@app.route('/login', methods=['get', 'post'])
def login():
    # 获取通过url请求传参的数据
    print("进来了")
    msg = request.values.get('msg')
    print(msg)
    if msg=="1":
        messg="000"
    else:
        messg="2222"
    print( json.dumps({"ms":messg}, ensure_ascii=False))
    return json.dumps({"ms":messg}, ensure_ascii=False)

#返回三元组得分与三元组正确与否
@app.route('/triple_correct', methods=['get', 'post'])
def triple_correct():
    transe.eval()
    thre = {}
    with open("./relation_thr.txt", "r",encoding="utf-8") as f:
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
    dist = float(transe(valid_data_vec)[0])
    r_thr=float((thre[str(r)][1]).replace("\n",""))
    judje= "知识正确" if dist<=r_thr else "知识错误"
    #处理与判断
    #形成最终
    # print( json.dumps({"ms":messg}, ensure_ascii=False))
    return json.dumps({"judje":judje,"dist":dist,"r_thr":r_thr}, ensure_ascii=False)



#返回关系阈值信息
@app.route('/threshold', methods=['get', 'post'])
def threshold():
    thre={}
    with open("./relation_thr.txt", "r",encoding="utf-8") as f:
        for l in f:
            msg=l.split("\t")
            thre[msg[0]]=[msg[1],msg[2]]
    return json.dumps({"ms":thre}, ensure_ascii=False)


#返回所有预测三元组  输出预测结果
@app.route('/allcorrect', methods=['get', 'post'])
def allcorrect():
    # 获取通过url请求传参的数据
    print("进来了")
    msg = request.values.get('msg')
    print(msg)
    if msg=="1":
        messg="000"
    else:
        messg="2222"
    print( json.dumps({"ms":messg}, ensure_ascii=False))
    return json.dumps({"ms":messg}, ensure_ascii=False)


#host="0.0.0.0" 代表谁都可以访问，可以加nginx处理
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=60015, debug=True)  ###指定端口、host设为0.0.0.0代表不管几个网卡，任何ip都可以访问
