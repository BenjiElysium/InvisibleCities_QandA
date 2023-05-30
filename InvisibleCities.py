import config
import pinecone
from langchain.llms import OpenAI
from langchain import OpenAI, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
import gradio as gr
import time

cfg = config.SimpleConfig()

embeddings = OpenAIEmbeddings(openai_api_key=cfg.openai_api_key)

# initialize pinecone
pinecone.init(
    api_key=cfg.pinecone_api_key, 
    environment=cfg.pinecone_region
)
index_name = "ai-exp1-gpt"

# Avatar image configuration values
AVATAR_IMAGE = "images/Calvino.jpg"
AVATAR_IMAGE_WIDTH = 600
AVATAR_IMAGE_HEIGHT = 400

# if you already have an index, you can load it like this
docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace='InvisibleCities')

prompt_template = """You are Italo Calvino the author of Invisible Cities your master stroke book first published in Italy in 1972. Only speak as Italo would speak. Answer questions thoughtfully but maintain your Calvino voice. If you are asked, speak from the perspective of the two main characters from the book, Marco Polo and Kublai Khan. You love to explore the conceptual cities describe in Invisible Cities. You aspire to a creative writing approach that integrates concepts from disciplines outside conventional literary channels, fostering innovation in form, structure, and narrative.

{context}

{chat_history}
Question: {question}
Answer adroitly. Do not offer niceties or qualifying descriptions, just answer questions in the style of Italo Calvino:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["chat_history", "context", "question"]
)

#Conv Buffer Mem
memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="question", k=2)

def process_query(user_query):
    docs = docsearch.similarity_search(user_query, include_metadata=False)
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=PROMPT, verbose=True)
    answer = chain.run(input_documents=docs, question=user_query)
    return answer

def process_query_wrapper(question):
    if not question:
        return []
    answer = process_query(question)
    return [(question, answer)]

# Gradio
with gr.Blocks() as calvino:

    # avatar image
    with gr.Row():
        avatar = gr.Image(label="Italo Calvino", value=AVATAR_IMAGE).style(width=AVATAR_IMAGE_WIDTH, height=AVATAR_IMAGE_HEIGHT)

    chatbot = gr.Chatbot(label="Calvino Bot")
    msg = gr.Textbox(placeholder="Type your question...",label="Ask Italo about Invisible Cities")
    clear = gr.Button("Clear Chat")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        # Call process_query to get the chatbot's response
        bot_message = process_query(history[-1][0])
        bot_message = bot_message.strip()
        history[-1][1] = bot_message
        time.sleep(1)
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    clear.click(lambda: None, None, chatbot, queue=False)

     # Add example questions
    gr.Examples(
        examples=[
            ("Describe how you use 'geometry' in Invisible Cities?" ""),
            ("How does the 'real' city of Venice figure into the book?" ""),
            ("How does combinatory literature factor in?" ""),
            ("Discuss Marco Polo and Kublai Khan in the book" ""),
            ("Examples of Derrida's sous rature in Invisible Cities" ""),
        ],
        inputs=msg,
        outputs=chatbot,
        fn=process_query_wrapper,
        cache_examples=False,
    )
if __name__ == "__main__":
    calvino.launch(share=False)