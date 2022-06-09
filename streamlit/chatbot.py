import streamlit as st
from streamlit_chat import message
import pandas as pd
import tensorflow as tf
from transformers import TFGPT2LMHeadModel
from transformers import AutoTokenizer

# BERT------------------------------------------
from transformers import BertTokenizerFast, TFBertModel
# import tensorflow as tf
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import Loss

from scipy.spatial.distance import cosine # 유사도
import numpy as np

#----------------------------------------------------------#
##### BERT #####
# 버트 모델 틀
class TFBertForSequenceClassification(tf.keras.Model):
    def __init__(self, model_name):
        super(TFBertForSequenceClassification, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        # 모든 가능한 분류를 위해 85를 지정했으나, 필요시 줄일 수 있음
        self.classifier = tf.keras.layers.Dense(85,
                                                kernel_initializer=TruncatedNormal(0.02),
                                                name='classifier')

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = outputs[1] # 문장 임베딩
        pred = self.classifier(cls_token)

        return pred

# BERT tokenizer
tokenizer_bert = BertTokenizerFast.from_pretrained("klue/bert-base", truncation_side='left')

# BERT 모델 불러오기
save_bert = './0516_simple_+val.h5-1'
m = TFBertForSequenceClassification("klue/bert-base")
chkpt = tf.train.Checkpoint(model=m)
local_device_option = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
chkpt.restore(save_bert, options=local_device_option)
m_bert = m

#----------------------------------------------------------#
##### GPT-2  #####
# GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token = '</pad>')

# GPT-2 model
model = TFGPT2LMHeadModel.from_pretrained('model_220420_3E_b8_all.h5')

#----------------------------------------------------------#

# from IPython.display import display

from scipy.spatial.distance import cosine, euclidean
import numpy as np
import pandas as pd


### BERT 사용 함수
def return_label_byBERT (sentence):
  input_sent = '[CLS] '+sentence.strip()+ ' [SEP]'
  xs = tokenizer_bert([input_sent], truncation=True, padding=True, max_length=512)
  keys = ['input_ids', 'attention_mask', 'token_type_ids']
  input_X=[]
  for k in keys:
      input_X.append(np.array(xs[k]))
  predScores = m_bert.predict(input_X)
  return predScores



# GPT 생성 후 마지막에 생성된 불완전하게 생성된 미종료 문장 제거
def cut_sent (sent):
    sen = sent.strip()
    if ((sen[-1] != '.') & (sen[-1] != '?') & (sen[-1] != '!')):
        findlist = [sen.rfind('.'), sen.rfind('?'), sen.rfind('!')]
        cut_index = max(findlist)
        sen = sen[:cut_index+1]
    return sen



 # 85개의 스코어를 6개의 라벨에 따라 나눠주는 함수
def target2list(arr):
    return [arr[:4], arr[4:6], arr[6:18], arr[18:21], arr[21:27], arr[27:]]



# 유사도 계산 함수
def calcSim(sim, simWeights):
    if simWeights: 
        simWeights = np.array(simWeights)/sum(simWeights)
        sim = sim*simWeights
        return np.sum(sim)
    return np.mean(sim)

#
def softmax(scores):
    return np.exp(scores)/np.exp(scores).sum()

#
import pickle
with open('decorders.pkl', 'rb') as f:
    decoders = pickle.load(f)

targetcols = ['연령', '성별', '상황키워드', '신체질환', '감정_대분류', '감정_소분류']
def simiarity(predScores, simWithPreds=True, printPreds=True, simWeights=None, method='cosine'):
    preds = pd.DataFrame(columns=targetcols, index=['Q', 'QnA'])
    for col, arr in zip(targetcols, target2list(predScores.T)):
        preds[col] = np.argmax(arr, 0)
        preds[col] = preds[col].map(decoders[col])
    if simWeights:
        preds = preds.append(pd.Series(simWeights, index=targetcols, name='weight'))
    else:
        preds = preds.append(pd.Series([1]*6, index=targetcols, name='weight'))
    # is equal? robust.
    if simWithPreds:
        sim = (preds.values[0] == preds.values[1]).astype('int')
        preds = preds.append(pd.Series(sim, index=targetcols, name='sim'))

    else:
        sim = {}
        for col, arr in zip(targetcols, target2list(predScores.T)):
            # similar direction of score vector space. stable
            if method == 'cosine':
                sim[col] = 1 - cosine(arr.T[0], arr.T[1]) # cos(theta)
            elif method == 'probcosine':
                arrleft, arrright = softmax(arr.T[0]), softmax(arr.T[1])
                sim[col] = 1 - cosine(arrleft, arrright) # cos(theta)
            # how near distance of score vector space. unstable
            else: # method == 'euclidian':
                sim[col] = 1 / euclidean(arr.T[0], arr.T[1])
        sim = pd.Series(sim, name='sim')
        preds = preds.append(sim)
 #   if printPreds: display(preds)
    return calcSim(sim, simWeights)



# 유사도에 적용될 가중치 생성함수
def sim_weight(gpt_label,method='defalut'):
  # ★target2list 함수 필요함
  if method == 'defalut':
    Weights = [1,1,1,1,1,1]
    return Weights

  elif method == 'maxarr':
    maxarr = []
    for arr in target2list(gpt_label[0]): 
      maxarr.append(np.max(arr, 0))
    return maxarr

  elif method == 'prob':
    prob_w = []
    for arr in target2list(gpt_label.T):
      arrsoft= softmax(arr.T[0])
      prob_w.extend(arrsoft)
    arr6=[]
    for arr in target2list(prob_w):
      arr6.append(np.max(arr, 0))
    return arr6



# 3문장 이상 생성될시, 섞여서 나오는 Q 자르는
def catch_questionMark (outputT):
  out_len = outputT.replace('?','.').replace('!','.').count('.') # 문장길이.

  def findIdx(outputT, value):
      n = -1
      result = []
      while True:
          if outputT[n+1:].count(value) == 0:
              break
          n += outputT[n+1:].index(value) + 1
          result.append(n)
      return result

  if (out_len == 3)&(outputT.count('?') == 1):     # 3문장 and '?' = 1개
    if (outputT[-1]=='?'): # c? , b자름
      tmp = outputT[(outputT.find('.')+1):(outputT.rfind('.')+1)]
      outputT = outputT.replace(tmp,"")
    elif (outputT.find('.') < outputT.find('?')):# b?, c자름
      outputT = outputT[:(outputT.find('?')+1)]
    elif (outputT.find('.') > outputT.find('?')):# a?
      outputT = outputT[:(outputT.find('?')+1)]

  elif out_len >= 3: # 길이 3개 이상
    if outputT.count('?') == 0:
      outputT = outputT[:(outputT.find('.')+1)]

    elif outputT.count('?') == 1:
      if (out_len >= 5): # 5문장 이상 = 첫문장만 가져옴
        if (outputT.find('.') < outputT.find('?')):
          outputT = outputT[:(outputT.find('?')+1)]
          outputT = outputT[:(outputT.find('.')+1)]
        else: 
          outputT = outputT[:(outputT.find('.')+1)]
          outputT = outputT[:(outputT.find('?')+1)]

      elif (outputT.find('.') > outputT.find('?')):
        outputT = outputT[:(outputT.find('?')+1)] # ?가 처음 = a?b.c. = 뒤를 다 버림

      elif ( (outputT.find('.') < outputT.find('?')) & (outputT.find('?') < outputT.find('.',outputT.find('.')+1)) ): # ?가 두번째 세번쨰가. (a.b?c.d)  = ?뒤를 다버림
        outputT = outputT[:(outputT.find('?')+1)]

      elif outputT.count('?') ==1 and outputT.count('.') ==3 and findIdx(outputT,'.')[2] > findIdx(outputT,'?')[0]:# a.b.c?d. 
        new = outputT.replace(outputT[findIdx(outputT,'.')[0]+1:findIdx(outputT,'.')[1]+1],'')
        outputT = new[:findIdx(new,'?')[0]+1]

      elif (outputT[-1]=='?'): #  ?가 네번째  = a.b.c. d?=  사이랑 뒤를자름 
        tmp = outputT[(outputT.find('.')+1):(outputT.rfind('.')+1)]
        outputT = outputT.replace(tmp," ")

      
    elif outputT.count('?') > 1:
      outputT = outputT[:(outputT.find('.')+1)]

  return outputT



# 쳇봇 대화 생성 함성
def return_answer_by_chatbot2(user_text):
    sent = '<usr>' + user_text + '<sys>'
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent)
    input_ids = tf.convert_to_tensor([input_ids])
    output = model.generate(input_ids, max_length=100, do_sample=True, top_k=10, num_return_sequences=10)

    sent_10 =[]
    # sentence 10 
    for i in range(len(output)):
        sentence = tokenizer.decode(output[i].numpy().tolist())
        chatbot_response = sentence.split('<sys>')[1].replace('</s>', '').replace('<pad>','').replace('<unk>','')
        sent_10.append(chatbot_response)
        # print(chatbot_response)

    input_text = return_label_byBERT(user_text)
    # 입력문장에 따라 값으로 가중치 (기본1 = defalt, 최고스코어 = maxarr, 확률 = prob) #0512
    similarity_weight = sim_weight(input_text,method='prob')

    # gpt가 공백으로 문장생성된 경우 삭제 #0513 : 선택된문장은 적용안됨(이건 maxlen문제로봄)
    [sent_10.remove(sent_10[i]) for i,x in enumerate(sent_10) if x.strip() == '']

    cosine_lst=[]
    for sent in sent_10:
        gpt_label = return_label_byBERT(sent)
        predScores = np.vstack([input_text,gpt_label])
        cosine_vl = simiarity(predScores, False, False, simWeights = similarity_weight,method='cosine')
        # cosine_vl = simiarity(predScores, False, True, [1, .1, 5, .01, 1, .1])
        # cosine_vl = 1 - cosine(input_text, gpt_label) # 1에서 
        cosine_lst.append(cosine_vl)
    maxidx = np.argmax(cosine_lst)
    a = cut_sent(sent_10[maxidx])
    b = catch_questionMark(a)
    return b

#
def conv(input_sp):
  inputT = input_sp
  # global saveSentList
  # print(saveSentList)
  saveSentList=st.session_state['saveSentList']
  saveSentList.append(inputT)

  if len(saveSentList) > 3:
    saveSentList = saveSentList[2:] #
  inputT = ' '.join(saveSentList) # 해봐야알것같음 '.'여부

  outputT = return_answer_by_chatbot2(inputT) # gpt-2 문장생성함수
  saveSentList.append(outputT)
  st.session_state['saveSentList']=saveSentList

  return outputT

@st.cache(allow_output_mutation=True)
def cached_model():
    model = TFGPT2LMHeadModel.from_pretrained('GPTmodel_well_3E_b8.h5')
    return model



model = cached_model()

# -------------출력----------------#
st.header('심리상담챗봇')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []   # 내가 입력한 내용 세션에 저장해놓기. 세션 실행 할 때마다 초기화 안되게.

# 입력문장 저장 리스트
if 'saveSentList' not in st.session_state:
    st.session_state['saveSentList'] = []
# saveSentList = []

# 텍스트박스 폼 만들기.
with st.form('form', clear_on_submit=True): # 입력 할 때마다 채팅창 클리어
    user_input = st.text_input('당신: ', '')
    submitted = st.form_submit_button('전송')   # 전송 버튼 누르면 입력한 인풋 내용 자동으로 지워진다.

if submitted and user_input:
    answer = conv(user_input)
    st.session_state.past.append(user_input)    # past 세션에 입력 내용 저장
    st.session_state.generated.append(answer)

# 말하는 내용이 메세지로!
for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')
# print(saveSentList)
print(st.session_state['past'])
print(st.session_state['saveSentList'])
# streamlit run chatbot.py