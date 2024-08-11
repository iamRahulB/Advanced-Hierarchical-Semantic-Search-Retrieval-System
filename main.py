import streamlit as st
from utils import FileSaver
from src.pdf_parser import TextProcessor
from src.models.clustering_and_semantic_search import ClusteringAndSemanticSearch
from src.models.gemini import GeminiLLM

# Initialize session state
if 'run' not in st.session_state:
    st.session_state.run = False

st.title('Advanced Hierarchical Semantic Search Retrieval System')

# URL input
url = st.text_input("Paste full link of PDF here")

# FileSaver object
filesaver_object = FileSaver()

@st.cache_data
def download_and_process_pdf(url):
    if filesaver_object.download_pdf(url):
        text_processor = TextProcessor()
        cleaned_text = text_processor.process_pdf('uploaded_files/downloaded.pdf')
        return cleaned_text
    else:
        raise ValueError("Failed to download the PDF. Please check the URL.")

if url:
    try:
        cleaned_text = download_and_process_pdf(url)
        st.success("File saved and processed successfully!")
        st.session_state.cleaned_text = cleaned_text
        st.session_state.run = True
    except ValueError as e:
        st.error(str(e))

if st.session_state.run:
    # Display the cleaned text
    # st.write("\n".join(st.session_state.cleaned_text[:1]))
    
    # Initialize the clustering and semantic search class
    clustering_search = ClusteringAndSemanticSearch()
    
    # Encode the text
    sentence_embeddings = clustering_search.encode_text(st.session_state.cleaned_text)
    
    # Perform clustering
    clustering_search.perform_clustering()
    
    # Query input and button displayed together
    query = st.text_input('Enter your query:')
    if query and st.button('Run Query'):
        # Find related sentences
        most_similar_sentence,related_sentences = clustering_search.find_related_sentences(query, st.session_state.cleaned_text)
        
        # Use related sentences as context
        context = " ".join(related_sentences[0:5])
        
        # Initialize the Gemini LLM
        gemini_llm = GeminiLLM()

        print(context)
        
        # Answer query using Gemini LLM
        answer = gemini_llm.run_extraction_chain(query, context)

        st.write(f"Semantic search result: \n\n{most_similar_sentence}\n\n")

        st.write(f"Sentence Hierarchy: \n\n {related_sentences[0:2]}\n\n")

        st.write(f"Gemini Answer: \n\n {answer}")
    
    st.session_state.run = False
