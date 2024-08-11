import langchain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage

class GeminiLLM:
    def __init__(self, model_name='gemini-1.5-flash', temperature=0.9):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    def generate_response(self, prompt):
        message = HumanMessage(content=prompt)
        response = self.llm([message])
        return response[0].content

    def run_extraction_chain(self, query, context):
        template = """
        You are an AI model designed to answer questions based on the given context. Please focus on the context provided and ensure your answers are relevant and accurate.

Context:{context}

\n\n

Question: {query}

Answer:
        """
        prompt_template = PromptTemplate(
            template=template,
            input_variables=["query", "context"]
        )
        extraction_chain = LLMChain(llm=self.llm, prompt=prompt_template)
        result = extraction_chain.run(query=query, context=context)
        return result
