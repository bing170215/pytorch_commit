import torch
import json
import numpy as np
import re
import datetime
import time
VER       = 12
#DATA_SIZE = 90661
DATA_SIZE = 512
B_SZ      = 64
K         = 20
CODE_VEC_PATH = 'code_vec.txt'


class ReadData:
    def load_data(self, version, load_difftext=True, load_msgtext=True, load_diff=True, load_msg=True,
                  load_variable=True, load_word2index=True):
        # java load data from disk 从磁盘中加载数据
        if load_difftext:
            self.difftext = json.load(open('data4CopynetV3/difftextV{}.json'.format(version)))
        if load_msgtext:
            self.msgtext = json.load(open('data4CopynetV3/msgtextV{}.json'.format(version)))
        if load_diff:
            self.difftoken = json.load(open('data4CopynetV3/difftokenV{}.json'.format(version)))
            self.diffmark = json.load(open('data4CopynetV3/diffmarkV{}.json'.format(version)))
            self.diffatt = json.load(open('data4CopynetV3/diffattV{}.json'.format(version)))
        if load_msg:
            self.msg = json.load(open('data4CopynetV3/msgV{}.json'.format(version)))
        if load_variable:
            self.variable = json.load(open('data4CopynetV3/variableV{}.json'.format(version)))
        if load_word2index:
            self.word2index = json.load(open('data4CopynetV3/word2indexV{}.json'.format(version)))
            self.genmask = json.load(open('data4CopynetV3/genmaskV{}.json'.format(version)))
            self.copymask = json.load(open('data4CopynetV3/copymaskV{}.json'.format(version)))
            self.difftoken_start = int(json.load(open('data4CopynetV3/numV{}.json'.format(version))))

    # gen_tensor2：@return: d_mark, d_word, d_attr
    # @d_mark: 直接从diffmarkV12.json中复制相应数量过来
    # @d_word: 和@d_attr: 对应difftoken和diffattr中的内容，并使用variable和word2index转化为向量进行存储
    def get_code_attr(self, start, end, diff_len=200, attr_num=5):
        length = end - start
        diff   = self.difftoken[start: end]  # 更新代码分词
        diff_m = self.diffmark [start: end]  # 代码变更情况（2不变1删除3添加）
        diff_a = self.diffatt  [start: end]  # 二次分词
        va     = self.variable [start: end]  # 标识符和占位符对应情况

        d_mark = np.zeros([length, diff_len])
        d_word = np.zeros([length, diff_len])
        d_attr = np.zeros([length, diff_len, attr_num])

        for i, (j, k, m, n) in enumerate(zip(diff, diff_m, va, diff_a)):
            # 对于第i个commit，diff[i], diff_m[i], diff_a[i]记录这次的代码变更情况
            for idx, (dt, dm, da) in enumerate(zip(j, k, n)):  # diff, diff_m, diff_a
                d_mark[i, idx] = dm             # 第i次commit的第idx个字段增删标记为dm
                dt = m[dt] if dt in m else dt   # 将标识符替换为占位符，若不存在这样的占位符则保留不变
                dn = self.word2index[dt] if dt in self.word2index else self.word2index['<unkd>']  # 替换为数字标记，若不是常用字则用<unkd>的标记
                d_word[i, idx] = dn             # 第i次commit的第idx个字段内容为dn
                for idx2, a in enumerate(da):   # 判断二次分词内容
                    if idx2 >= attr_num:        # 只检索前attr_num个词
                                                # 若词语不够attr_num，不会自动补足
                        break
                    # 第i次commit的第idx个字段的第idx2个分词为a
                    d_attr[i, idx, idx2] = self.word2index[a] if a in self.word2index else self.word2index['<unkd>']
        return d_mark, d_word, d_attr

    def get_commit_vec(self, start, end, commit, msg_len=20):
        # {"added": "add", "fixed": "fix", "removed": "remove", "adding": "add", "fixing": "fix", "removing": "remove"}
        lemmatization = json.load(open('lemmatization.json'))
        length = end - start
        msg = commit  # commit信息
        va = self.variable[start: end]  # 标识符和占位符对应情况
        mg = np.zeros([length, msg_len + 1])
        for i, m in enumerate(va):
            # 对于第i个commit，msg[i]记录commit信息
            mg[i, 0] = 1  # 固定为1
            # 遍历commit信息
            for idx, c in enumerate(msg):
                c = m[c] if c in m else c.lower()  # 将标识符替换为占位符，若不存在则转换为小写
                c = lemmatization[c] if c in lemmatization else c  # 特定文本替换
                c0 = self.word2index[c] if c in self.word2index else self.word2index['<unkm>']  # 替换为数字标记，若不是常用字则用<unkm>的标记
                # difftoken_start = 10130
                c0 = self.word2index['<unkm>'] if c0 >= self.difftoken_start else c0  # 大于difftoken_start的替换为<unkm>的标记，否则不变
                mg[i, idx + 1] = c0  # 记录这个字段的内容标记
        return mg

def get_code_vec(code_learner):
    ######################################
    # 读取数据
    ######################################
    dataset = ReadData()
    dataset.load_data(VER)
    mark, word, attr = dataset.get_code_attr(0, DATA_SIZE)

    cur = 0
    code_vec = None
    while(cur < DATA_SIZE):
        ######################################
        # 计算向量
        ######################################
        m = torch.from_numpy(mark[cur:min(DATA_SIZE, cur+B_SZ)]).long().to(device)
        w = torch.from_numpy(word[cur:min(DATA_SIZE, cur+B_SZ)]).long().to(device)
        a = torch.from_numpy(attr[cur:min(DATA_SIZE, cur+B_SZ)]).long().to(device)
        cur += B_SZ

        print('batch:', str(int(cur/B_SZ)) + '/' + str(int((DATA_SIZE+B_SZ-1)/B_SZ)))
        vec = code_learner(m, w, a)  # 预测code_vec

        ######################################
        # 储存向量
        ######################################
        vec = vec.detach().cpu().numpy().reshape((-1, 512)) # 512????
        code_vec = np.concatenate((code_vec, vec), axis=0) if code_vec is not None else vec

    ######################################
    # 保存到文件
    ######################################
    # np.savetxt('code_vec\code_vec' + str(int(cur/B_SZ)) + '.txt', code_vec)
    np.savetxt(CODE_VEC_PATH, code_vec)
    print('code_vec已保存至' + CODE_VEC_PATH)
    return

def get_top_k(dataset, commit, k, code_vecs, commit_learner, class_learner,device,cur=0,DATA_SIZE=DATA_SIZE):


    ######################################
    # 计算向量
    ######################################
    #cur = 0
    cur_temp=cur
    simi = None
    while(cur < DATA_SIZE):
        ######################################
        # 得到预计算好的code_vec
        ######################################
        if cur_temp!=0:
            code_vec = code_vecs[cur-cur_temp:min(cur-cur_temp + B_SZ, DATA_SIZE-cur_temp)]
        else:

            code_vec = code_vecs[cur:min(cur+B_SZ, DATA_SIZE)]
        # print("code_vec: ", code_vec)
        start1 = time.time()
        ######################################
        # 计算commit_vec
        ######################################
        #mg = dataset.get_commit_vec(cur, min(cur+B_SZ, DATA_SIZE), commit)
        mg = dataset.get_commit_vec(cur, min(cur + B_SZ, DATA_SIZE), commit)
        mg = torch.from_numpy(mg).long().to(device)
        commit_vec = commit_learner(mg)
        # print("commit_vec: ", commit_vec)


        ######################################
        # 计算相似度
        ######################################
        code_v = torch.from_numpy(code_vec).float().to(device)
        commit_v = commit_vec
        class_vec = class_learner(code_v, commit_v)
        # print('class_vec: ', class_vec)
        # pos = class_vec.detach().cpu().numpy().reshape((-1, 2))
        # # print('pos: ', pos)
        # pos = pos[:,1]
        pos = class_vec.detach().cpu().numpy()
        #pos = pos[:,0]
        #print('pos: ', pos)
        end1 = time.time()

       # print('time for this batch:' + str(round(end1 - start1, 2)))


        ######################################
        # 保存处理
        ######################################
        simi = np.concatenate((simi, pos), axis=0) if simi is not None else pos
        # print(simi.shape)
        # print(simi)

        cur += B_SZ
        #print('batch:', str(int(cur / B_SZ)) + '/' + str(int((DATA_SIZE + B_SZ - 1) / B_SZ)))

    ######################################
    # 排序返回
    ######################################
    simi = torch.from_numpy(simi)
    _, pre = simi.topk(k)
    # print('_: ', _)
    # print('pre: ', pre)
    return pre.numpy() # 最大的k个值的下标

def split_msg(msgtext):
    # to split msg in case build function are wrong
    msgs = list()
    pattern = re.compile(r'\w+')

    msg = pattern.findall(msgtext)
    msg = [j for j in msg if j != '' and not j.isspace()]
    msgs.append(msg)
    return msgs

def get_codes(top_ids):
    ######################################
    # 读取数据
    ######################################
    dataset = ReadData()
    dataset.load_data(VER)
    codes=np.array(dataset.difftoken)[top_ids]

    return codes


def get_commits(dataset, top_ids):
    ######################################
    # 读取数据
    ######################################
    commits = np.array(dataset.msgtext)[top_ids]

    return commits






if __name__ == '__main__':


    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ######################################
    # 加载模型
    ######################################
    main_path = './models/'
    #
    # code_path = main_path + 'code_Model_NEG5_cossimi.pkl'
    # code_learner = torch.load(code_path)
    # code_learner.eval()
    # print('code_learner加载成功')
    # get_code_vec(code_learner)                                        # 预训练仓库中的code_vec

    commit_path = main_path + 'commit_Model_NEG5_cossimi.pkl'
    commit_learner = torch.load(commit_path)
    commit_learner.eval()
    print('commit_learner加载成功')
    class_path = main_path + 'class_Model_NEG5_cossimi.pkl'
    class_learner = torch.load(class_path)
    class_learner.eval()
    print('class_learner加载成功')


    dataset = ReadData()
    dataset.load_data(VER)


    print('加载预训练code_vec...')
    code_vecs = np.loadtxt(CODE_VEC_PATH)
    print('code_vec加载成功')

    start = time.time()


    ######################################
    # 查找top-k
    ######################################
    # commit = input('commit: ').strip()                                # 输入查找关键词commit
    #commit = "Handle the getter and setter case"
    #commit="Fix bug where class should delegate to setDetails method - not set the details directly."
    commit="don't use SupportCode for generating setDisplayedMnemonicIndex()"
    #commit = re.split(r'[ ,;.=!@#$%^&*()_+-=\'\"`~?:<>{}\[\]]', commit) # 分词
    commit = split_msg(commit)
    print(commit)
    #
    time_top_k = time.time()
    print("start top k: ", time_top_k - start)
    #top_ids = get_top_k(dataset, commit[0], K, code_vecs, commit_learner, class_learner)
    top_ids =get_top_k(dataset, commit[0], K, code_vecs, commit_learner, class_learner,device)# 根据训练好的code_vec匹配相似度最高的k个代码段的id
    print("time for top k: ", time.time() - time_top_k)

    # codes = get_codes(list(top_ids))
    time_commits = time.time()
    commits = get_commits(dataset, list(top_ids))
    print("time for commits: ", time.time() - time_commits)
    # print(codes)
    print(commits)
    end = time.time()
    print('time for this search:' + str(round(end - start, 2)))

    num_of_choosen = 100
    choosen_idx = np.random.choice(7500, size=num_of_choosen, replace=False)


    #定义一个变量用来存储命中的个数
    sum=0
    # for idx,commit in enumerate(dataset.msgtext[:DATA_SIZE]):
    #     commit = split_msg(commit)
    #     top_ids = get_top_k(dataset, commit[0], K, code_vecs, commit_learner, class_learner)
    #     if idx in top_ids:
    #         sum = sum +1

    for idx in choosen_idx:
        commit = split_msg(dataset.msgtext[idx])
        top_ids = get_top_k(dataset, commit[0], K, code_vecs, commit_learner, class_learner)
        if idx in top_ids:
            sum = sum +1

    accu = float(sum/num_of_choosen)
    print('topK='+str(K))
    print('命中数='+str(sum))
    print('准确率='+str(accu))






