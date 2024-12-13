import os
import args
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from sentence_transformers import SentenceTransformer
    

loaders = {
    '.pdf': PyMuPDFLoader,
    '.doc': UnstructuredWordDocumentLoader,
    '.docx': UnstructuredWordDocumentLoader,
    '.xlsx': UnstructuredExcelLoader,
    '.pptx': UnstructuredPowerPointLoader
}

    
def create_directory_loader(file_type, directory_path):
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=loaders[file_type],
    )


def load_docs():
    root_path = args.root_path
    file_path = os.path.join(root_path, args.data_extra_path)
    file_suffix = ['.pdf', '.doc', '.docx', '.pptx', '.xlsx']

    dir_loarders = [create_directory_loader(v, file_path) for v in file_suffix]
    documents = [loader.load() for loader in dir_loarders] # 得到二维列表，每个元素是一个 doc/pdf/...的所有文件
    docs = [file for file_list in documents for file in file_list]
    return docs


def load_llm_embedding():
    llm = ChatOpenAI(model_name=args.llm_model_name, base_url=args.llm_base_url)
    embedding = SentenceTransformer(args.embedding_model_name, device='cuda')
    return llm, embedding