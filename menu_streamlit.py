import streamlit as st


if 'role' not in st.session_state :
    st.session_state.role = 'patient'
    


def authenticated_menu():
    # Show a navigation menu for authenticated users
    
    
    st.sidebar.page_link("app_streamlit.py", label="API key 재등록하기")
            
    st.sidebar.page_link("pages/result_opinion.py", label="검사 소견 자동 작성")
    st.sidebar.page_link("pages/result_opinion_new.py", label="새로운 검사 소견 자동 작성 및 소견 요약")
    st.sidebar.page_link("pages/overall_opinion.py", label="종합 소견 자동 작성")
    st.sidebar.page_link("pages/overall_opinion_english.py", label="종합 소견 자동 작성 (영문버전)")
    
            
    if 'patient_info' in st.session_state:
        st.sidebar.write(st.session_state.patient_info)
    if "result_finalreport" not in st.session_state:
        st.sidebar.page_link("pages/result_submit.py", label="검진 결과 상담하기")
    if "result_finalreport" in st.session_state:
        st.sidebar.page_link("pages/consultation_chatbot.py", label="검진 결과 상담하기")    
    #st.sidebar.write(st.session_state)
    
    st.sidebar.page_link("pages/validation.py", label="validation")

def unauthenticated_menu():
    # Show a navigation menu for unauthenticated users
    st.sidebar.page_link("app_streamlit.py", label="API key 등록하기")


def menu():
    # Determine if a user is logged in or not, then show the correct
    # navigation menu
    with st.sidebar:
        if "openai_api_key" not in st.session_state or st.session_state.openai_api_key == '':
            unauthenticated_menu()
            return
    authenticated_menu()


def menu_with_redirect():
    # Redirect users to the main page if not logged in, otherwise continue to
    # render the navigation menu
    
    if "openai_api_key" not in st.session_state or st.session_state.openai_api_key == '':
        st.switch_page("app_streamlit.py")
    menu()