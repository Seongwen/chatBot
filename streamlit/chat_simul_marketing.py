import streamlit as st
from streamlit_chat import message
import time


st.markdown("<h2 style='text-align: center; color: violet;'>감성챗봇 프로젝트</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>위안이 필요한 당신과 프로방스에센셜이 함께 합니다.</h4>", unsafe_allow_html=True)
st.markdown("<h7 style='text-align: center; color: black;'>챗봇과 대화를 나눠보세요. 당신의 감정상태에 맞는 아로마 오일을 추천 해 드립니다.</h7>", unsafe_allow_html=True)

with st.form('form', clear_on_submit=True): # 입력 할 때마다 채팅창 클리어
    user_input = st.text_input('대화를 입력해 주세요 : ', '')
    submitted = st.form_submit_button('전송')   # 전송 버튼 누르면 입력한 인풋 내용 자동으로 지워진다.

time.sleep(7)
message("안녕", is_user=True) 
time.sleep(1)
with st.spinner('입력 중...'):
    time.sleep(4)
message("안녕하세요.") 
time.sleep(3)
message("오늘 직장에서 깨져서 너무 우울해.", is_user=True) 
time.sleep(1)
with st.spinner('입력 중...'):
    time.sleep(4)
message("오늘 직장에서의 일로 기분이 안좋으시군요. 좀 더 자세히 말해줄 수 있나요?") 
time.sleep(4)
message("내 실수로 수량을 잘못 입력했어. 너무 속상해서 계속 그 생각만 하게 돼.", is_user=True) 
time.sleep(1)
with st.spinner('입력 중...'):
    time.sleep(4)
message("다음엔 더 잘 할 수 있을거에요.") 

time.sleep(4)
with st.spinner('지금 당신의 감성에 맞는 아로마 오일을 추출 중입니다...'):
    time.sleep(8)

my_bar = st.progress(0)
# for percent_complete in range(100):
#      time.sleep(0.1)
#      my_bar.progress(percent_complete + 1)
st.balloons()
st.success('심신을 진정시켜 줄 백단향은 어떠세요? 심리적으로 긴장과 불안, 스트레스를 완화시키고 마음을 편안하게 진정시켜 줄거에요.')