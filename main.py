import streamlit as st
from langchain_helper import process_urls, create_vector_db, get_qa_chain

st.title("Web Data Retrieval Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

if process_url_clicked:
    main_placeholder.text("Data Processing...Started...✅✅✅")
    docs = process_urls(urls)
    main_placeholder.text("Creating Vector Database...✅✅✅")
    create_vector_db(docs)

query = main_placeholder.text_input("Question: ")

if query:
    chain = get_qa_chain()
    result = chain.invoke({"question": query}, return_only_outputs=True)
    answer = result['answer'].replace("FINAL ANSWER:", "").strip()
    st.header("Answer")
    st.write(answer)

    # Display sources, if available
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources_list:
            st.write(source)