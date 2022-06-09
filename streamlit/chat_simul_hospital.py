import streamlit as st
from streamlit_chat import message
import time


st.markdown("<h2 style='text-align: center; color: violet;'> 병원 예약 도우미</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>NPNG 종합병원</h4>", unsafe_allow_html=True)
st.markdown("<h8 style='text-align: center; color: black;'>*수집된 환자의 정보는 진료 외의 목적으로는 활용되지 않습니다.</h8>", unsafe_allow_html=True)
with st.form('form1', clear_on_submit=False): # 입력 할 때마다 채팅창 클리어
    name = st.text_input('이름', '고영희')
    tel = st.text_input('전화번호 마지막 4자리', '2739')
    submitted = st.form_submit_button('입력')   # 전송 버튼 누르면 입력한 인풋 내용 자동으로 지워진다.

with st.form('form2', clear_on_submit=True): # 입력 할 때마다 채팅창 클리어
    user_input = st.text_input('현재 몸 상태를 구체적으로 알려주세요. : ', '')
    submitted = st.form_submit_button('전송')   # 전송 버튼 누르면 입력한 인풋 내용 자동으로 지워진다.

time.sleep(10)
message("안녕하세요, 고영희님. 어디가 불편해서 오셨어요?") 
time.sleep(3)
message("목 언저리에 뭐가 만져지는데, 어느 과로 가야할지 모르겠어요.", is_user=True) 
time.sleep(1)
with st.spinner('입력 중...'):
    time.sleep(4)
message("목 언저리에 뭐가 만져지시는군요. 좀 더 자세히 말씀해 주시겠어요?") 
time.sleep(3)
message("만졌을 때 아프진 않은데, 종양이 아닐까 걱정돼요.", is_user=True) 
time.sleep(1)
with st.spinner('입력 중...'):
    time.sleep(4)
message("증상이 오래 되셨나요?") 
time.sleep(5)
message("2주 정도는 된 것 같은데요.", is_user=True) 
time.sleep(1)
with st.spinner('입력 중...'):
    time.sleep(4)
message("추가적인 증상이 있으신가요?") 
time.sleep(5)
message("요즘 좀 피곤한 것 말고는 딱히 없어요.", is_user=True) 
time.sleep(4)
with st.spinner('지금 환자분의 상태를 분석하고 있습니다...'):
    time.sleep(8)

my_bar = st.progress(0)
for percent_complete in range(100):
     time.sleep(0.01)
     my_bar.progress(percent_complete + 1)
st.success('3층 "내분비과"로 내원해주세요. 예약을 도와드리겠습니다.')