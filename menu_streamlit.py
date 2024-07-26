import streamlit as st


if 'role' not in st.session_state :
    st.session_state.role = 'patient'
    


def authenticated_menu():
    # Show a navigation menu for authenticated users
    
    
    st.sidebar.page_link("app_streamlit.py", label="API key 재등록하기")
            
    st.sidebar.page_link("pages/result_opinion.py", label="검사 소견 자동 작성")
    st.sidebar.page_link("pages/overall_opinion.py", label="종합 소견 자동 작성")
            
    if 'patient_info' in st.session_state:
        st.sidebar.write(st.session_state.patient_info)
    #st.sidebar.write(st.session_state)

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