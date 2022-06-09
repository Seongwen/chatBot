import streamlit as st
from streamlit_chat import message
import time


st.markdown("<h2 style='text-align: center; color: black;'> 학교 상담을 돕는 챗봇</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: black;'>세모고등학교</h4>", unsafe_allow_html=True)
st.markdown("<h8 style='text-align: center; color: grey;'>*학생과의 대화의 내용은 비밀이 보장됩니다.</h8>", unsafe_allow_html=True)

with st.form('form', clear_on_submit=True): # 입력 할 때마다 채팅창 클리어
    user_input = st.text_input('대화를 입력해 주세요. : ', '')
    submitted = st.form_submit_button('전송')   # 전송 버튼 누르면 입력한 인풋 내용 자동으로 지워진다.
time.sleep(8)
message("안녕. 무슨 생각해?") 
time.sleep(4)
message("친구들이 다 나를 싫어해.", is_user=True) 
time.sleep(1)
with st.spinner('입력 중...'):
    time.sleep(4)
message("많이 힘들었겠다. 언제 가장 힘든 것 같아?") 
time.sleep(4)
message("밥 먹으러 갈 때나 축구를 할 때 나를 따돌려.", is_user=True) 
time.sleep(1)
with st.spinner('입력 중...'):
    time.sleep(4)
message("친구들이 따돌릴 때 어떤 기분이 들어?") 
time.sleep(5)
message("마음이 답답해. 왜 그러는지 친구들한테 이유를 물어보고 싶어.", is_user=True) 
time.sleep(1)
with st.spinner('입력 중...'):
    time.sleep(5)
message("어떻게 하면 친구들에게 물어볼 수 있을까?") 
time.sleep(6)
message("모르겠어. 누가 방법을 좀 알려줬으면 좋겠는데.", is_user=True) 
time.sleep(5)
with st.spinner(''):
    time.sleep(8)
st.success('"학교 사회복지사 오은영 선생님이 도움을 주고 싶어하세요. 연결해 드릴까요?')