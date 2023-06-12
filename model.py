import streamlit as st

# Streamlit app code
def main():
    st.title("My Streamlit App")
    st.write("Welcome to my Streamlit app!")

    name = st.text_input("Enter your name")
    st.write(f"Hello, {name}!")

if __name__ == "__main__":
    main()
