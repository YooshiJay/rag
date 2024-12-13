import os
import args
import time
from retriever import MyRetriever
from evaluate import eval
from load import load_llm_embedding, load_docs
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


def main():
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['OPENAI_API_KEY'] = args.openai_api_key
    os.environ["HTTP_PROXY"] = args.http_proxy
    os.environ["HTTPS_PROXY"] = args.http_proxy

    llm, embedding = load_llm_embedding()
    docs = load_docs()

    start = time.time()
    print("开始构建检索器")
    retriever = MyRetriever(llm, embedding, docs)
    end = time.time()
    print("检索器构建完成，用时：", end-start)

    # template = "你是一名专门负责回答规章制度的助手，使用以下检索到的上下文来回答问题。" \
    #             "如果你不知道答案，请直接说不知道。答案请保持简洁明了。" \
    #             "\n\n上下文: {context} "

    template = "你是一名专门负责回答规章制度的助手，使用以下检索到的上下文来回答问题。" \
                "题目为选择题或多选题，请仅输出选项。如：A, B, C, D" \
                "\n\n上下文: {context} "

    prompt = ChatPromptTemplate([
        ('system', template),
        ('human', '\n\n问题: {query} ')],
        input_variables=["context", "query"]
    )
    chain = create_stuff_documents_chain(llm, prompt)
    eval(chain, retriever)


if __name__ == "__main__":
    
    main()