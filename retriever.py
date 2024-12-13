import faiss
import numpy as np
from typing import List, Optional, Sequence, Any
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


sep = "<SEP>"

DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""你是一个AI语言模型助手。 你的任务是生成3个不同版本的用户问题，
    进而可以从向量数据库中检索相关文档。通过从不同角度生成用户问题，
    你的目标是帮助用户克服基于距离的相似性搜索的一些局限性。
    请将这些替代问题用""" + sep + """分隔。原始问题：{question}""",
)


class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""
    
    def has_content(self, x: str) -> bool:
        return x.strip()
    
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("sep")
        return list(filter(self.has_content, lines))  # Remove empty lines


class MyRetriever():
    
    def __init__(self, llm, embedding, docs):
        self.llm = llm
        self.embedding = embedding
        self.docs = docs
        self.template = DEFAULT_QUERY_PROMPT
        self.output_parser = LineListOutputParser()
        
        self.full_docs = None
        self.sub_docs = None
        self.parent_ids = None
        
        self.index_gpu = None
        self.all_query = None
        
        self.create_parent_document()
        self.create_faiss_index()
        

    @staticmethod
    def flatten_with_priority(arr: Sequence[Sequence[Any]]) -> Sequence[Any]:
        result = arr.T.flatten()
        return result
    
    
    @staticmethod
    def remove_duplicate(arr):
        '''去重且保留顺序'''
        seen = set()
        unique_list = []

        for item in arr:
            if item not in seen:
                unique_list.append(item)
                seen.add(item)

        return np.array(unique_list)
    
        
    def create_parent_document(self):
        # 分割文件
        parent_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "。", "；", "，", " ", ""], chunk_size=300, chunk_overlap=50)
        child_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "。", "；", "，", " ", ""], chunk_size=100, chunk_overlap=20)

        # 父文件分割
        self.full_docs = parent_splitter.split_documents(self.docs)

        # 关联子文件
        self.sub_docs = []
        self.parent_ids = []
        for _id, _doc in enumerate(self.full_docs):
            sd = child_splitter.split_documents([_doc])
            self.parent_ids.extend([_id] * len(sd))
            self.sub_docs.extend(sd)

        self.parent_ids = np.array(self.parent_ids)
        
        
    def create_faiss_index(self):
        pure_docs = [file.page_content for file in self.sub_docs]
        datas_embedding = self.embedding.encode(pure_docs)
        index_cpu = faiss.IndexFlatL2(datas_embedding.shape[1])
        index_with_ids = faiss.IndexIDMap(index_cpu)

        res = faiss.StandardGpuResources()
        self.index_gpu = faiss.index_cpu_to_gpu(res, 0, index_with_ids)
        self.index_gpu.add_with_ids(datas_embedding, np.arange(self.parent_ids.shape[0]))
        
    
    def invoke(self, query: str) -> Any:
        llm_chain = self.template | self.llm | self.output_parser
        self.all_query = llm_chain.invoke({"question": query})
        query_embedding = self.embedding.encode(self.all_query)
        Distance, Index = self.index_gpu.search(query_embedding, 4)
        Index_reorder = self.flatten_with_priority(Index)
        
        unique_ids = self.remove_duplicate(self.parent_ids[Index_reorder])
        contexts = [self.full_docs[_id] for _id in unique_ids]
        return contexts        
    