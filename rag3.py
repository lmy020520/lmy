# 优化版本2
import pandas as pd
import numpy as np
import torch
import requests
import re
import json
import os
import jieba
from docx import Document
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

class DocumentRAGTestingSystem:
    def __init__(self, knowledge_base_path: str, model_path: str, deepseek_api_key: str, 
                 paragraph_split_length: int = 300, use_advanced_splitting: bool = True,
                 use_reranking: bool = True, use_hybrid_search: bool = True):
        """
        初始化文档RAG测试系统
        
        Parameters:
            knowledge_base_path: 知识库Word文档路径(.docx)
            model_path: 本地嵌入模型路径  
            deepseek_api_key: DeepSeek API密钥
            paragraph_split_length: 将文档分割成段落的最大长度(字符数)
            use_advanced_splitting: 是否使用高级文档分割
            use_reranking: 是否使用重排序  
            use_hybrid_search: 是否使用混合搜索策略
        """
        self.knowledge_base_path = knowledge_base_path
        self.model_path = model_path
        self.api_key = deepseek_api_key
        self.paragraph_split_length = paragraph_split_length
        self.use_advanced_splitting = use_advanced_splitting
        self.use_reranking = use_reranking
        self.use_hybrid_search = use_hybrid_search
        
        self.knowledge_base = None
        self.embedding_model = None
        self.vector_db = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # 高级配置
        self.overlap_size = 100  # 段落重叠大小
        self.reranking_top_k = 10  # 重排序前检索的文档数
        self.final_top_k = 3  # 最终返回的文档数
        
        self.initialize_knowledge_base()
    
    def parse_word_document(self, file_path: str) -> List[Dict]:
        """Parse the Word document and split the content into paragraphs"""
        try:
            doc = Document(file_path)
            all_text = []
            
            # 收集所有段落和它们的文本
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    all_text.append(text)
            
            if not all_text:
                raise ValueError("No usable text content found in the document")
            
            if self.use_advanced_splitting:
                return self.advanced_document_splitting(all_text)
            else:
                return self.simple_document_splitting(all_text)
            
        except Exception as e:
            raise ValueError(f"Failed to parse Word document: {str(e)}")
    
    def simple_document_splitting(self, text_list: List[str]) -> List[Dict]:
        """简单的文档分割，基于段落长度"""
        paragraphs_list = []
        current_paragraph = ""
        
        for text in text_list:
            # If the current paragraph plus new text does not exceed the maximum length, merge them
            if len(current_paragraph) + len(text) < self.paragraph_split_length:
                current_paragraph += " " + text if current_paragraph else text
            else:
                # Save the current paragraph and start a new one
                if current_paragraph:
                    paragraphs_list.append({"content": current_paragraph})
                current_paragraph = text
        
        # Add the last paragraph
        if current_paragraph:
            paragraphs_list.append({"content": current_paragraph})
        
        return paragraphs_list
    
    def advanced_document_splitting(self, text_list: List[str]) -> List[Dict]:
        """使用Jieba分词+规则实现高级文档分割"""
        full_text = ' '.join(text_list)
        sentences = []
        current_sentence = []

        # 初始化Jieba（确保分词效果更准确）
        jieba.initialize()
        end_punctuations = ['。', '！', '？', '；', '…', '，', '.', '!', '?', ';']
        # 使用Jieba分词并基于标点符号分句
        for word in jieba.cut(full_text):
            current_sentence.append(word)
            if word in end_punctuations:  # 常见中文句子结束符号
                sentence = ''.join(current_sentence).strip()
                if sentence:
                    sentences.append(sentence)
                current_sentence = []
        
        # 处理剩余的文本
        if current_sentence:
            remaining_sentence = ''.join(current_sentence).strip()
            if remaining_sentence:
                sentences.append(remaining_sentence)
        
        # 基于滑动窗口的段落创建，带有重叠
        paragraphs_list = []
        current_paragraph = ""
        overlap_text = ""
        dynamic_paragraph_length = self.paragraph_split_length * (len(text_list) / 1000) if len(text_list) > 1000 else self.paragraph_split_length

        for i, sentence in enumerate(sentences):
            # 如果当前段落加上新句子不超过最大长度，添加到当前段落
            if len(current_paragraph) + len(sentence) < dynamic_paragraph_length:
                current_paragraph += " " + sentence if current_paragraph else sentence
            else:
                # 保存当前段落
                if current_paragraph:
                    paragraphs_list.append({"content": current_paragraph})
                
                # 创建带有重叠的新段落
                current_paragraph = overlap_text + " " + sentence if overlap_text else sentence
                
                # 更新重叠文本（取前3句作为重叠）
                overlap_text = " ".join(sentences[max(0, i - 3):i])
        
        # 添加最后一个段落
        if current_paragraph:
            paragraphs_list.append({"content": current_paragraph})
        
        return paragraphs_list
    
    def initialize_knowledge_base(self):
        """Load the document knowledge base and initialize the embedding model"""
        try:
            # Load the knowledge base Word document
            if not os.path.exists(self.knowledge_base_path):
                raise FileNotFoundError(f"Knowledge base file not found: {self.knowledge_base_path}")
            
            if not self.knowledge_base_path.endswith('.docx'):
                raise ValueError("The knowledge base file must be in .docx format")
            
            self.knowledge_base = self.parse_word_document(self.knowledge_base_path)
            
            # Load the local embedding model
            print(f"Loading embedding model ({self.model_path})...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if not os.path.exists(self.model_path):
                self.model_path = 'all-mpnet-base-v2'
                
            self.embedding_model = SentenceTransformer(
                model_name_or_path=self.model_path,
                device=device
            )
            
            # Generate vectors for document paragraphs
            print("Generating vectors for document paragraphs...")
            paragraph_contents = [entry['content'] for entry in self.knowledge_base]
            self.vector_db = self.embedding_model.encode(
                paragraph_contents, 
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            
            # 初始化TF-IDF向量器(用于混合搜索)
            if self.use_hybrid_search:
                print("Initializing TF-IDF vectorizer...")
                self.tfidf_vectorizer = TfidfVectorizer()
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(paragraph_contents)
            
            print(f"Knowledge base initialization completed, loaded {len(self.knowledge_base)} document paragraphs")
            
        except Exception as exception:
            print(f"Initialization failed: {str(exception)}")
            raise
    
    def retrieve_relevant_paragraphs(self, query: str, num_results: int = 3) -> List[Dict]:
        """Retrieve the most relevant document paragraphs using hybrid search and reranking"""
        if not self.knowledge_base or self.vector_db is None:
            raise RuntimeError("Knowledge base not initialized")
        
        if self.use_hybrid_search:
            # 混合搜索策略
            vector_results = self._retrieve_by_embedding(query, self.reranking_top_k * 2)
            tfidf_results = self._retrieve_by_tfidf(query, self.reranking_top_k * 2)
            
            # 合并结果并去重
            combined_results = {}
            for result in vector_results + tfidf_results:
                content = result["content"]
                if content not in combined_results:
                    combined_results[content] = result
                else:
                    # 取最高相似度
                    combined_results[content]["similarity"] = max(
                        combined_results[content]["similarity"], 
                        result["similarity"]
                    )
            
            # 按相似度排序
            candidates = sorted(combined_results.values(), key=lambda x: x["similarity"], reverse=True)
            candidates = candidates[:self.reranking_top_k]
        else:
            # 仅使用嵌入搜索
            candidates = self._retrieve_by_embedding(query, self.reranking_top_k)
        
        # 应用重排序
        if self.use_reranking and len(candidates) > num_results:
            final_results = self._rerank_results(query, candidates, num_results)
        else:
            final_results = candidates[:num_results]
        
        return final_results
    
    def _retrieve_by_embedding(self, query: str, num_results: int) -> List[Dict]:
        """使用嵌入向量检索相关段落"""
        # Generate query vector
        query_vector = self.embedding_model.encode(
            query, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        # Calculate similarity
        cosine_similarity = util.pytorch_cos_sim(query_vector, self.vector_db)[0]
        euclidean_distance = torch.norm(self.vector_db - query_vector, dim=1)
        combined_score = cosine_similarity - 0.1 * euclidean_distance
        similarity_results = torch.topk(combined_score, k=min(num_results, len(self.knowledge_base)))
        
        # Format results
        result_list = []
        for score, index in zip(similarity_results[0], similarity_results[1]):
            entry = self.knowledge_base[index]
            result_list.append({
                "content": entry["content"],
                "similarity": float(score),
                "index": int(index)
            })
        
        return result_list
    
    def _retrieve_by_tfidf(self, query: str, num_results: int) -> List[Dict]:
        """使用TF-IDF检索相关段落"""
        query_tfidf = self.tfidf_vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        
        # 获取前N个最相似的段落索引
        top_indices = cosine_similarities.argsort()[-min(num_results, len(cosine_similarities)):][::-1]
        
        result_list = []
        for index in top_indices:
            result_list.append({
                "content": self.knowledge_base[index]["content"],
                "similarity": float(cosine_similarities[index]),
                "index": int(index)
            })
        
        return result_list
    
    def _rerank_results(self, query: str, candidates: List[Dict], num_results: int) -> List[Dict]:
        """对候选结果进行重排序，考虑多样性和相关性"""
        if len(candidates) <= num_results:
            return candidates
        
        # 创建候选段落的嵌入向量
        candidate_contents = [candidate["content"] for candidate in candidates]
        candidate_vectors = self.embedding_model.encode(
            candidate_contents, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        # 计算候选段落之间的相似度矩阵
        similarity_matrix = util.pytorch_cos_sim(candidate_vectors, candidate_vectors)
        
        # 基于MMR(最大边际相关性)算法进行重排序
        selected_indices = []
        candidate_indices = list(range(len(candidates)))
        
        # 首先选择与查询最相关的段落
        query_vector = self.embedding_model.encode(
            query, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        query_similarities = util.pytorch_cos_sim(query_vector, candidate_vectors)[0]
        most_relevant_idx = torch.argmax(query_similarities).item()
        
        selected_indices.append(most_relevant_idx)
        candidate_indices.remove(most_relevant_idx)
        
        # 继续选择段落，直到达到所需数量
        lambda_param = 0.8  # 控制相关性和多样性的平衡
        
        while len(selected_indices) < min(num_results, len(candidates)) and candidate_indices:
            best_score = -1
            best_idx = -1
            
            for idx in candidate_indices:
                # 计算与查询的相关性
                relevance_score = query_similarities[idx].item()
                
                # 计算与已选段落的多样性(最大相似度)
                diversity_scores = [similarity_matrix[idx][selected_idx].item() for selected_idx in selected_indices]
                diversity_score = max(diversity_scores) if diversity_scores else 0

                length_score = len(candidates[idx]["content"]) / 1000
                
                # MMR公式: λ * 相关性 - (1-λ) * 多样性 + 0.1 * 长度得分
                mmr_score = lambda_param * relevance_score - (1 - lambda_param) * diversity_score + 0.1 * length_score
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
        
        # 返回重排序后的结果
        final_results = [candidates[idx] for idx in selected_indices]
        return final_results
    
    def call_deepseek_api(self, prompt: str) -> str:
        """Call the DeepSeek API to generate a response"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        request_body = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 800
        }
        
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=request_body
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exception:
            raise RuntimeError(f"API request failed: {str(exception)}")
    
    def generate_prompt(self, question: str, relevant_paragraphs: list) -> str:
        """Generate a prompt with relevant knowledge"""
        background_knowledge = "\n".join(
            f"段落内容: {entry['content']}\n(相关度:{entry['similarity']:.2f})" 
            for entry in relevant_paragraphs
        )

        example_question = "示例问题: 什么是生物学评价？"
        example_answer = "示例答案: 生物学评价是对医疗器械或材料与生物体相互作用的评估。"
        
        return f"""基于以下文档内容回答问题：

===== 相关文档段落 =====
{background_knowledge}

===== 示例问题与答案 =====
{example_question}
{example_answer}

===== 待回答问题 =====
{question}

请根据上述文档内容直接给出最准确的答案，不要解释。"""
    
    def evaluate_qa_effect(self, test_questions: List[str]) -> pd.DataFrame:
        """
        Evaluate the Q&A effectiveness
        
        Parameters:
            test_questions: List of questions to test
        
        Returns:
            A DataFrame containing the test results, including:
            - Test question
            - RAG system answer
            - Retrieved relevant paragraphs
        """
        result_table = []
        
        for question in test_questions:
            try:
                # 1. Retrieve relevant paragraphs
                relevant_paragraphs = self.retrieve_relevant_paragraphs(question)
                
                # 2. Generate prompt and get RAG answer
                prompt = self.generate_prompt(question, relevant_paragraphs)
                rag_answer = self.call_deepseek_api(prompt)
                
                # 4. Record result
                result_table.append({
                    "测试问题": question,
                    "RAG答案": rag_answer,
                    "相关段落": relevant_paragraphs
                })
                
            except Exception as exception:
                result_table.append({
                    "测试问题": question,
                    "RAG答案": f"错误: {str(exception)}",
                    "相关段落": []
                })
        
        return pd.DataFrame(result_table)
    
# Example usage
if __name__ == "__main__":
    # Configuration parameters
    knowledge_base_path = "/home/dockeruser/lmy/二级医疗器械/河北省官网资料要求.docx"  # Replace with your Word document path
    model_path = "/home/dockeruser/lmy/Model/model"
    deepseek_key = "sk-883d825876464ab6966616a3ae887953"
    
    # Initialize the testing system
    tester = DocumentRAGTestingSystem(
        knowledge_base_path=knowledge_base_path,
        model_path=model_path,
        deepseek_api_key=deepseek_key,
        paragraph_split_length=300  # Adjust paragraph length as needed
    )
    
    # Prepare test questions and corresponding standard answers
    test_questions = [


"是否所有医疗器械均需提交‘辐射安全研究’？",
"是否必须委托第三方机构进行产品检验？",
"是否所有产品均需提供‘使用稳定性’研究？",
"是否所有软件均需进行‘网络安全研究’？",
"是否必须开展动物试验？",
"是否所有申报产品均需提供‘不良事件’历史？",
"是否所有产品均需包含‘禁忌证’说明？",
"是否必须提交‘主文档授权信’？",
"是否所有产品均需进行‘临床评价’？",
"是否所有原材料均需提供来源证明？"
    ]
    
    test_results = tester.evaluate_qa_effect(test_questions)
    
    # Print results
    print("\n测试结果:")
    print(test_results[['测试问题', 'RAG答案']])
    
    # Save detailed results
    test_results.to_excel("文档问答测试结果10.xlsx", index=False)
    print("详细结果已保存到 文档问答测试结果.xlsx")
    
    # 输出每个问题的相关段落（如果需要）
    for _, row in test_results.iterrows():
        print(f"\n问题: {row['测试问题']}")
        print(f"RAG答案: {row['RAG答案']}")
        print("相关段落:")
        for para in row['相关段落']:
            print(f" - 段落: {para['content']} (相关度:{para['similarity']:.2f})")