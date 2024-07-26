import streamlit as st


if 'role' not in st.session_state :
    st.session_state.role = 'patient'
    


def authenticated_menu():
    # Show a navigation menu for authenticated users
    
    
    st.sidebar.page_link("app_streamlit.py", label="API key 재등록하기")
            
    st.sidebar.page_link("pages/patient_info.py", label="환자정보 등록")
    st.sidebar.page_link("pages/chat_assistant.py", label="CHAT ASSISTANT")
            
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
        if "openai_api_key_psscode" not in st.session_state or st.session_state.openai_api_key_psscode == '':
            unauthenticated_menu()
            return
    authenticated_menu()


def menu_with_redirect():
    # Redirect users to the main page if not logged in, otherwise continue to
    # render the navigation menu
    
    if "openai_api_key_psscode" not in st.session_state or st.session_state.openai_api_key_psscode == '':
        st.switch_page("app_streamlit.py")
    menu()