# 감성분석 기반 심리상담 AI 챗봇
**"일상이 된 비대면…소통부재·고립"**

코로나19 사태의 장기화, 거리두기의 일상화로 재택근무와 원격수업 등이 일상으로 자리 잡았습니다.   
2022년에는 비대면 문화에 기반한 서비스가 훨씬 세련되고 정교화될 것이지만, 이런 상황이 불러올 부작용으로 대면 소통이 줄면서 사회 전체가 움츠러들고 사람들이 고립되어간다 생각했습니다. 

## 목적
코로나 사태로 사람을 못만나면서 우울감이 늘어나고있습니다. 우울감은 챗봇으로도 급감한다는 결과의 기사를 접했습니다. 그래서 사용자의 감정을 인식해 상호작용을 하는 챗봇 프로젝트를 진행했습니다.

보통 챗봇 심리상담으로 얻을 수 있는 기대효과는 다음과 같습니다.  
- 좋은 접근성, 24시간연결
- 비대면
- 장기적으로 비용절감

이러한 이유들에도 불구하고 현재, 소비자가 챗봇에 만족하지 못한 이유를 알아보았습니다 (출처 챗봇 소비자 만족도, Spicesworks)   
<img src="https://user-images.githubusercontent.com/97740175/173789066-ec6c3ed4-c590-4fe8-8408-7a493315c808.png" width="60%" height="60%"></img>

이런 문제에 직면했을때 소비자가 챗봇 서비스를 이용하기를 포기한다는 점을 눈여겨보고, 이러한 부분을 개선하기위해 아래의 3가지를 목표로 진행했습니다. 
1. GPT-2 모델을 사용하여, 고정되지 않은 다양한 형태의 문장 생성.   
2. BERT 모델을 사용하여, **문장 맥락 파악**과 사용자 특징, 감정, 상황 등을 반영.   
3. Q,A,Q형식으로 대화 내용을 기억하여 맥락파악   



## 기간
2021.03.31 - 2021.05.19 (8 주)


## 팀 구성
본인 외 2인

## 팀 내 역할
Python을 활용한 데이터 결측치 처리, 라벨 정수화   
훈련된 BERT모델을 사용하여 스코어 출력 기능 구현   
GPT모델로 생성된 문장 중 길이제한으로 미완성 문장 제거 기능 구현   
사용자의 말투가 섞여서 생성된 경우, 케이스에 따라 문장 부분 삭제 기능 구현   
입력문장에 구두점이 없을 경우, 끝에 "." 붙여서 전달하는 기능 구현   
대화를 Q+A+Q 형식으로 저장하며 이전 대화 반영 기능 구현   


## 사용 기술
python, koGPT-2, klueBERT, streamlit


## 개발 과정 
- 앞서 진행한 3개의 프로젝트와 다르게 마지막 프로젝트는 처음 진행한 장기 프로젝트로, 하루하루 일정을 기록하고 전체를 관리하며 진행했습니다.   
<img src="https://user-images.githubusercontent.com/97740175/173689837-329990e9-6eec-4dbb-ba8d-0472862c8dbc.png" width="90%" height="90%"></img>

### 데이터 분석

#### 데이터 소개

1. 감성대화 말뭉치 데이터 (AIhub 2021.06)
* 27만개의 말뭉치 데이터는 주로 우울증 관련 언어로 이루어져있습니다. 최대 4번의 연속적인 대화가 이어집니다. 
* 분류에 특화되어있습니다. 4개의 연령, 성별, 12개의 상황키워드, 3개의 만성질환, 6개의 감정 대분류로 분류되어있습니다. 6개의 감정 대분류는 하위 58가지 감정 소분류로 다시 분류됩니다.

<img src="https://user-images.githubusercontent.com/96275852/173796803-1460ff68-8985-429a-b78c-0a67e66a880b.png" width="100%" height="100%">

2. 웰니스 대화 스크립트
* 세브란스 상담 데이터를 기반으로 구축한 정신 상담 데이터셋입니다.
* 359개 대화의도에 대한 5,232개 사용자 발화와 1,023개 챗봇 발화로 이루어져있습니다.
* 이 데이터는 보다 생동감 있는 챗봇 발화를 위해 GPT 모델 학습에만 추가했습니다.
<img src="https://user-images.githubusercontent.com/96275852/173799606-fc0a5180-e53b-4c34-9fb4-060e3b302cc1.png" width="100%" height="100%">

#### 데이터 전처리

<img src="https://user-images.githubusercontent.com/96275852/173795115-7723b301-d863-4c92-9462-7a0a0b9be8ea.png" width="70%" height="70%">

##### 정수화
BERT 모델 학습을 위한 라벨링을 진행했습니다.  
4개의 연령, 성별, 12개의 상황키워드, 3개의 신체질환, 6개의 감정 대분류, 하위 58가지 감정 소분류를 라벨링했습니다.

```
labeling_cols = ['연령', '성별', '상황키워드', '신체질환', '감정_대분류',	'감정_소분류']
for label in labeling_col:
  temp_list = []
  temp_list = all_df2[label].unique().tolist()  # 임시 리스트에 고유값 넣기
  all_df2[label] = all_df2[label].map(lambda x :temp_list.index(x)) # 정수화
```
##### 결측치 제거
짝이 맞지 않는 문장은 결측처리 했습니다.

```
# 발화만있고 응답이 없거나 발화는 없는데 응답만 있는 대화쌍의 조건
Nan_cond = all_df2['사람문장4'].isnull() != all_df2['시스템응답4'].isnull()

# 조건이 해당되는 사람문장4를 Nan처리
all_df2.loc[(Nan_cond),'사람문장4'] = np.nan
all_df2.loc[(Nan_cond),'시스템응답4'] = np.nan
```
##### QAQ 합치기
*편의를 위해 사람문장은 Q, 시스템응답은 A로 표기하겠습니다.*   
최대 4번 연속되는 사용자와 챗봇의 대화를 아래와 같은 형태로 변경해주었습니다. 데이터 형태를 변경해줌으로써 이전에 챗봇과 나눈 대화 내용을 기억하게해, 대화에 연속성을 반영할 수 있습니다.     
<img src="https://user-images.githubusercontent.com/96275852/173801308-a4ce5fdc-3786-45e7-9d58-3ea65f92264f.png" width="30%" height="30%">

##### 토큰 삽입
QA를 구분하기 위해 둘 사이에 [SEP]토큰을 삽입합니다. BERT에 내장된 토큰과 동일한 토큰입니다.

##### 데이터 추가
보다 나은 성능을 위해 데이터를 추가했습니다. 두가지 모델의 용도에 맞게 데이터를 다르게 추가했습니다. 

* GPT 모델 : 문장 생성 모델입니다. 보다 생동감 있는 답변을 위해 웰니스 대화 스크립트를 추가했습니다.
* BERT 모델 : 분류를 위한 모델입니다. A 데이터는 Q와 동일한 라벨링이 되어있습니다. df에 Q+A열을 추가로 만들어 학습 데이터를 2배로 만들었습니다. Q와 Q+A열을 모두 학습시킴으로써 모델의 분류 성능을 높였습니다.
```
df['QnA'] = df['Q'] + ' [SEP] ' + df['A']
```

### 모델
#### 사용모델
> **GPT모델**    
> <img src="https://user-images.githubusercontent.com/97740175/173801316-e5273ee5-f34c-437c-b6e7-a6dbdbb00ff4.png" width="40%" height="40%"></img>   
> - 디코더를 활용하여 순차적 단어를 생성하는 모델입니다.
> - 챗봇의 다양한 대답 생성을 위해 사용하였습니다.
>
>##### KoGPT2 (skt/kogpt2-base-v2)
>skt/kogpt2-base-v2모델을 pretrained하여 사용하였습니다.   
>데이터를 Q(사용자의말)와 A(대답)형태로 하여, 훈련시켜 들어온 문장 뒤에 대답이 생성되도록 학습시켰습니다.     
>이과정에서, 학습되는 기존의 감성대화 말뭉치는 너무 경직된 말투라 생각되어 학습 데이터 셋에 웰니스 대화 스크립트 추가로 말투 개선을 의도했습니다.   
><img src="https://user-images.githubusercontent.com/97740175/173804067-fecef858-9b84-44f1-b299-dc182764de3d.png" width="42%" height="42%"></img>    
>자연어 생성 모델이기에 정답이 없어 Accuracy측정은 할 수 없었습니다. 후반에 직접 성능 검증을 했습니다.

</br>


> **BERT 모델**   
> <img src="https://user-images.githubusercontent.com/97740175/173801334-9d14a710-84d6-4003-87c8-72bf2c5418f5.png" width="40%" height="40%"></img>   
> - 인코더를 활용하여 양방향분석을 하는 모델입니다.
> - 생성된 문장의 분류를 위해 사용하였습니다.
>
>##### KlueBERT (klue/bert-base)
>klue/bert-base모델을 pretrained하여 사용하였습니다.   
>데이터는 카테고리 분류가 된 감성대화 말뭉치만 사용하였습니다.
>Q와, Q+A를 입력으로 하여 카테고리를 예측하는 형태로 모델을 훈련시켰습니다. y는 각 카테고리에 대한 85개의 스코어값입니다.    
><img src="https://user-images.githubusercontent.com/97740175/173806997-99e07b47-8568-4129-a693-a62735f91001.png" width="60%" height="60%"></img>    
>들어온 X(문장)을 6개 카테고리(하위 총 85개)로 분류, y와의 유사도를 통해 스코어를 계산합니다.    
>각 6개 카테고리에대한 Accuracy :    
><img src="https://user-images.githubusercontent.com/97740175/173806542-5830d553-7fa0-41ac-aa49-004f8f5badeb.png" width="70%" height="70%"></img> 


#### 모델 구조도
<img src="https://user-images.githubusercontent.com/97740175/173807655-c28376da-436f-40a0-aac8-299935ce82da.png" width="70%" height="70%"></img>    
1. 입력된 user text에 문장구분을 위한 토큰\<user>,\<sys>을 붙여 GPT-2 모델로 문장을 생성합니다. \<user>은 문장의 시작을, \<sys>끝을 표시하고 gpt모델은 \<sys>에 이어지는 문장을 생성합니다.
2. GPT-2모델에서 10개의 문장을 랜덤으로 생성합니다
3. 생성된 10개의 문장은 BERT모델로 라벨에 대한 각각 스코어 점수를 갖습니다.
4. 10문장에 대한 10개의 스코어 점수에 유사도 측정으로 가장 높은 점수를 받은 문장을 선택하여 후처리 후, 출력합니다.

#### 모듈 구조도
<img src="https://user-images.githubusercontent.com/96275852/173841150-c1d809c6-54d4-4d25-ab20-77047724ab20.png" width="75%" height="75%"></img>

* conv = 이전 대화 내용을 저장합니다. 매번 발화를 다시 시작할 필요 없습니다.
* return_answer_by_chatbot2 = 속에서 함수들이 작동해 출력 문장을 생성합니다.

#### 모델 성능비교
> 모델의 성능을 평가하기위한 두가지의 데이터셋을 만들었습니다.   
> - **test set 1** = 에폭, wellness 추가, 가중치 등의 변수를 비교하기 위해 사용한 데이터.   
>   - 감성대화말뭉치(50), 웰니스데이터(50), 인상한말(30), 감성대화에 없는 새로 만든 말(30)으로 총 입력문장 160개
> - **test set 2**  = 대화 저장 유무와  방식에 따른 성능을 비교하기 위해 만든 데이터. 
>   - 감성대화 말뭉치 데이터 속 네번의 연속된 대화가 이루어진 50개의 케이스
>   - 50개의 대화 케이스를 9종류의 대화저장 형태에 따라 데이터셋으로 만들었습니다.

**GPT-2** 생성 성능에 대한 테스트
- Epoch(E),  wellness(\_well) 유무   
test set1  1600 문장을 검토 후, 틀린 문장의 개수를 카운트했습니다.    
연속대화가 아닌, 단일 문장에 대한 생성 품질을 테스트했습니다.   
3에폭 wellness데이터가 포함된 모델의 성능이 가장 좋게나왔습니다.   

</br>

- 이전대화 정보가 반영되는지 
3epoch wellness데이터가 포함된 모델로 문장을 생성했습니다.   
test set2의 문장 검토 후, 틀린 문장의 개수를 카운트 했습니다.   
QAQ형식으로 입력문장을 넣었을 경우의 오답률이 가장 낮았습니다.   

</br>

- maxlen 
GPT모델의 문장 생성시 maxlen값을 설정합니다.    
maxlen은 입력 토큰 길이와 출력 토큰의 길이를 포함합니다.   
GPT모델로 입력을 할땐 토큰화진행후에 입력이 됩니다. 한 문장당 평균 18개의 토큰으로 이루어집니다.   

> - maxlen이 길 경우( maxlen = 100 ) :   
>   - 긴 만큼 많은 문장을 생성해야해 속도가 느려집니다.    
>   - 입력 문장이 짧을 경우 이상하게 긴 문장을 만들어 낼 확률이 높아집니다.   

> - maxlen이 짧을 경우( maxlen = 50 ):   
>   - 사용자의 말만 되풀이하는 천편일률적인 문장이 생성됩니다.   
>   - 새로 입력한 문장과 누적된 대화가 합쳐져 입력으로 들어가기 때문에 문장을 생성하다가 생성공간부족으로 도중에 끊겨버리는 현상이 발생합니다.    

</br>

**BERT** 생성 성능에 대한 테스트
- BERT 가중치   
맥락을 이해하는 챗봇을 만들기 위해 **상황(라벨)** 에 가중치 부여를 시도했습니다. 기본 값은 모두 1입니다.     
임의의 가중치 = {  연령: 1,  성별: 0.1,  상황: 5,  만성질환 0.01,  감정 대분류: 1,  감정 소분류: 0.1  }   
임의로 가중치를 주었을 때와 주지 않았을 때 accuracy 차이 비교해 보았을때, 가중치를 주지 않는 것이 결과가 좋았습니다.   
<img src="https://user-images.githubusercontent.com/97740175/173844747-75ac79b7-1645-4364-91e7-d974863253ce.png" width="70%" height="70%"></img>

</br>


- 유사도 함수
BERT모델로 계산된 85개의 라벨의 대한 스코어를 입력문장과 유사도 비교를 할때 cosine, euclidian 유사도 함수를 사용해 비교했습니다.  
(prob_  :  score 값을 양수로 만들기 위해서 logit 값을 지수 값으로 변환한 후,  softmax로 정규화한 값.)   
(count는 입력문장의 라벨과 일치한 개수를 카운트한 값입니다.)   

</br>

#### 성능 비교 - 결과
<img src="https://user-images.githubusercontent.com/97740175/173845896-14ed5178-c52d-46b9-bcbe-dadba003b106.png" width="50%" height="50%"></img>

- 3epoch, wellness가 포함된 데이터셋으로 학습하고 max len은 75인 GPT-2 모델이 가장 좋은 성능을 보였습니다.   
- 라벨별 학습가중치가 같은 BERT모델이가장 좋은 성능을 보였습니다.   
- 대화 저장 방식은 Q1+A1+Q2 방식이, 입력문장과 스코어의 유사도 가중치는 기본값인1, softmax로 정규화한 cosine인 prob_cosine을 사용했을때 가장 좋은 성능이었습니다.   


</br>

## 챗봇시연
<img src="https://user-images.githubusercontent.com/96275852/173818876-ac7ef388-98a9-4e50-8d5e-48fb97f7f83e.gif" width="70%" height="70%"></img>   
Streamlit을 사용해 서비스를 구현 했습니다. 청소년이 챗봇 상담을 한다는 가정 하에 대화 내용을 구성했습니다. 사용자의 감정과 상황에 적합한 답변을 하고있음을 볼 수 있습니다.    

</br>

## 서비스확장   
<img src="https://user-images.githubusercontent.com/96275852/173846993-3a1aadb0-c80b-4d26-ad92-6c9ed17140ff.png" width="70%" height="70%"></img>      
챗봇은 정보수집에 있어 목적이 표면에 드러나지 않기때문에 심리적 장벽이 낮다는 장점이 있습니다. 사용자와의 대화를 통해 성별, 연령, 심리 상태 등을 자연스럽게 파악해 이를 토대로 적절한 제품 추천이 가능합니다. 그 중에서도 저희 챗봇은 사용자와 나눈 대화를 분류 할 수 있다는 것이 가장 큰 특징 입니다. 6가지 감정 대분류를 사용자의 니즈로 가정했습니다.   


> - **기분에 맞는 에센셜 오일을 추천 해 주는 시스템입니다.**         
> [![Video Label](http://img.youtube.com/vi/uLR1RNqJ1Mw/0.jpg)](https://youtube.com/shorts/j20XOfTSWts?feature=share)  


> - **사전 문진 챗봇입니다.**  
> <img src="https://user-images.githubusercontent.com/97740175/173849048-652017f7-5178-4b52-9ed8-96975c34640a.png" width="60%" height="60%"></img>   
> 이 영역은 데이터가 의료 임상 관련으로 분류가 되어 있어야 하는 부분이라 현재 가진 데이터로는 힘들지만, 챗봇과 분류를 접목해서 구상가능한 서비스 중 하나로 가정해보았습니다.   
> 종합병원에서 앱으로 사전 문진을 한다고 가정하고 서비스를 구상했을때, 증상을 말하면 진료에 필요한 정보들을 식별해서 기록하고 분류한 다음 해당 과에 환자를 배정(환자입장에서는 예약)하는 것으로 마무리 되는 흐름으로 구성해보았습니다.   




</br>



<hr>

## 회고   
- 데이터를 제공하는 AIhub에는 분류 기준이 명확하게 기재되어있지 않았습니다. 의미 중첩으로 보이는 클래스들이 있었지만, 임의로 데이터를 수정하기에는 데이터의 변별력을 잃게될 것 같아 조심스러웠습니다. 감성말뭉치 데이터에서 이 부분을 명확히 확인하지 못한 점이 아쉽습니다.
- 다중 분류를 위한 BERT모델 선정하는 작업과 성능 비교를 위한 테스트 작업에서 많은 시간이 소요되었습니다. 유사도측정으로 문장을 선택했을때 더 좋은 문장이 있는데도 유사도가 가장높게 안나와 선택이 안되는 경우를 개선하고자, 각 라벨에 대해 다른 유사도를 주고싶었습니다. 하지만 시간 부족으로 인해 가중치를 다양하게 테스트 해보지 못해 아쉽습니다. 
- GPU가 없는 개인 컴퓨터에서 구동하기엔 속도가 느립니다. 챗봇 시연과 서비스 확장에서 보여드린 영상은 CPU환경에서 4배속을 한 영상입니다. 챗봇의 볼륨이 클수록 원활한 구동을 위해서 고사양의 서버가 필요한데, 시중의 챗봇 볼륨이 작은 이유는 고사양 서버를 빌리는데 드는 비용이 상당하기 때문이라는 피드백을 받았습니다. 다음 작업에는 이를 고려해, 보다 시장성을 반영한 챗봇으로 발전시키고 싶습니다. 
- GPT모델 학습중 batch를 8보다 크게 늘릴수가없었습니다. 이 GPU문제를 해결하고자 colab pro 유료 계정을 사용하였지만, 여전히 8까지가 한계였습니다. 그래서 BERT처럼 TPU환경에서 학습을 해보고자 하였지만, 원인모를 토큰 문제가 발생하여 프로젝트 기간상 포기하고 8배치로 가게되었습니다. 더 다양한 배치를 줘서 모델성능을 비교하지 못해 아쉬웠고 하드웨어 성능을 비롯한 작업 한계가 제한된 환경이 아쉬웠습니다.
- 본론 시작전의 일상대화를 어떻게 처리할까 의논을 많이했었습니다. 사전에 제작된 리스트를 사용해 인사말만 간단히 처리해줄수있는 기능을 구현하거나, 일상대화만 학습된 모델을 하나 더 사용하는 방법을 구상해보았으나, 정해진 시간상 구현을 하지못했습니다.
