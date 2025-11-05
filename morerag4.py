#FAISS精确向量检索（事实性问题）
#语义聚类检索（解释性问题）
#基于图的检索（推理性问题）
#混合加权检索（边界测试问题）
#多样性MMR检索（模糊查询）
#倒排索引检索（否定问题）
import pandas as pd
import numpy as np
import torch
import requests
import re
import os
import jieba
from docx import Document
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import faiss
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

class MultiTypeDocumentRAGSystem:
    def __init__(self, knowledge_base_path: str, model_path: str, deepseek_api_key: str, 
                 roberta_model_path: str, paragraph_split_length: int = 300, 
                 use_advanced_splitting: bool = True, use_reranking: bool = True, 
                 use_hybrid_search: bool = True):
        """
        初始化多类型文档RAG系统
        
        Parameters:
            knowledge_base_path: 知识库Word文档路径(.docx)
            model_path: 本地嵌入模型路径  
            deepseek_api_key: DeepSeek API密钥
            roberta_model_path: Chinese-RoBERTa-wwm-ext模型路径
            paragraph_split_length: 将文档分割成段落的最大长度(字符数)
            use_advanced_splitting: 是否使用高级文档分割
            use_reranking: 是否使用重排序  
            use_hybrid_search: 是否使用混合搜索策略
        """
        self.knowledge_base_path = knowledge_base_path
        self.model_path = model_path
        self.roberta_model_path = roberta_model_path
        self.api_key = deepseek_api_key
        self.paragraph_split_length = paragraph_split_length
        self.use_advanced_splitting = use_advanced_splitting
        self.use_reranking = use_reranking
        self.use_hybrid_search = use_hybrid_search
        
        self.knowledge_base = None
        self.embedding_model = None
        self.roberta_model = None
        self.roberta_tokenizer = None
        self.vector_db = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.faiss_index = None
        self.lsa_vectors = None
        
        # 高级配置
        self.overlap_size = 100  # 段落重叠大小
        self.reranking_top_k = 10  # 重排序前检索的文档数
        self.final_top_k = 3  # 最终返回的文档数
        
        # 问题类型定义
        self.question_types = {
            "factual": "事实性问题",
            "explanatory": "解释性问题", 
            "inferential": "推理性问题",
            "boundary": "边界测试问题",
            "ambiguous": "模糊查询",
            "negative": "否定问题"
        }
        
        # 检索方法定义
        self.retrieval_methods = {
            "factual": "faiss_exact_search",
            "explanatory": "semantic_cluster_search",
            "inferential": "graph_based_search",
            "boundary": "hybrid_weighted_search",
            "ambiguous": "diverse_mmr_search",
            "negative": "inverted_index_search"
        }
        
        self.initialize_knowledge_base()
        self.initialize_roberta_model()
    
    def initialize_roberta_model(self):
        """初始化Chinese-RoBERTa-wwm-ext模型"""
        try:
            print(f"Loading Chinese-RoBERTa-wwm-ext model from {self.roberta_model_path}...")
            self.roberta_tokenizer = BertTokenizer.from_pretrained(self.roberta_model_path)
            self.roberta_model = BertForSequenceClassification.from_pretrained(
                self.roberta_model_path, 
                num_labels=len(self.question_types)
            )
            
            # 设置模型为评估模式
            self.roberta_model.eval()
            
            # 如果有GPU，将模型移到GPU上
            if torch.cuda.is_available():
                self.roberta_model = self.roberta_model.cuda()
                
            print("Chinese-RoBERTa-wwm-ext model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load Chinese-RoBERTa-wwm-ext model: {str(e)}")
            raise
    
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
        
        # 使用Jieba分词并基于标点符号分句
        for word in jieba.cut(full_text):
            current_sentence.append(word)
            if word in ['。', '！', '？', '；', '…', '，']:  # 常见中文句子结束符号
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
        
        for i, sentence in enumerate(sentences):
            # 如果当前段落加上新句子不超过最大长度，添加到当前段落
            if len(current_paragraph) + len(sentence) < self.paragraph_split_length:
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
                raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
                
            self.embedding_model = SentenceTransformer(
                model_name_or_path=self.model_path,
                device=device
            )
            
            # Generate vectors for document paragraphs
            print("Generating vectors for document paragraphs...")
            paragraph_contents = [entry['content'] for entry in self.knowledge_base]
            embeddings = self.embedding_model.encode(
                paragraph_contents, 
                show_progress_bar=True,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            
            self.vector_db = torch.tensor(embeddings) if isinstance(embeddings, np.ndarray) else embeddings
            
            # 初始化TF-IDF向量器
            print("Initializing TF-IDF vectorizer...")
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(paragraph_contents)
            
            # 初始化FAISS索引（用于精确向量检索）
            print("Initializing FAISS index...")
            embedding_dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
            self.faiss_index.add(embeddings.astype('float32'))
            
            # 初始化LSA降维（用于语义聚类）
            print("Initializing LSA for semantic clustering...")
            n_components = min(50, self.tfidf_matrix.shape[1])
            self.lsa = TruncatedSVD(n_components=n_components, random_state=42)
            self.lsa_matrix = self.lsa.fit_transform(self.tfidf_matrix)
            self.lsa_matrix = Normalizer(copy=False).fit_transform(self.lsa_matrix)
            
            # 构建文档图（用于基于图的检索）
            print("Building document graph...")
            self.build_document_graph()
            
            print(f"Knowledge base initialization completed, loaded {len(self.knowledge_base)} document paragraphs")
            
        except Exception as exception:
            print(f"Initialization failed: {str(exception)}")
            raise

    def build_document_graph(self):
        """构建文档关联图"""
        # 计算文档间的相似度矩阵
        similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        # 创建图
        self.document_graph = nx.Graph()
        
        # 添加节点
        for i in range(len(self.knowledge_base)):
            self.document_graph.add_node(i, content=self.knowledge_base[i]['content'])
        
        # 添加边（只保留前20%的最强连接）
        threshold = np.percentile(similarity_matrix, 80)
        for i in range(len(self.knowledge_base)):
            for j in range(i + 1, len(self.knowledge_base)):
                if similarity_matrix[i, j] > threshold:
                    self.document_graph.add_edge(i, j, weight=similarity_matrix[i, j])

    def classify_question_type(self, question: str) -> str:
        """
        使用Chinese-RoBERTa-wwm-ext模型分类问题类型
        """
        try:
            # 如果模型未初始化，使用规则分类
            if self.roberta_model is None or self.roberta_tokenizer is None:
                return self._classify_by_rules(question)
            
            # 使用RoBERTa模型进行分类
            inputs = self.roberta_tokenizer(
                question, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            )
            
            # 将输入移到GPU（如果可用）
            if torch.cuda.is_available():
                inputs = {key: value.cuda() for key, value in inputs.items()}
            
            # 模型预测
            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
                predictions = softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(predictions, dim=1).item()
            
            # 映射预测结果到问题类型
            question_types_list = list(self.question_types.keys())
            if predicted_class < len(question_types_list):
                return question_types_list[predicted_class]
            else:
                return self._classify_by_rules(question)
                
        except Exception as e:
            print(f"RoBERTa classification failed: {str(e)}, falling back to rule-based classification")
            return self._classify_by_rules(question)
    
    def _classify_by_rules(self, question: str) -> str:
        """规则分类问题类型 - 备用方法"""
        question_lower = question.lower()
        
        # 规则匹配
        if any(word in question_lower for word in ["什么", "哪些", "何时", "谁", "多少", "是否"]):
            if "为什么" in question_lower or "原因" in question_lower or "如何" in question_lower:
                return "inferential"
            elif "不" in question_lower or "没" in question_lower or "无" in question_lower:
                return "negative"
            else:
                return "factual"
                
        elif "为什么" in question_lower or "原因" in question_lower or "如何" in question_lower:
            return "inferential"
            
        elif "什么是" in question_lower or "解释" in question_lower or "含义" in question_lower:
            return "explanatory"
            
        elif "哪些情况" in question_lower or "什么时候" in question_lower or "条件" in question_lower:
            return "boundary"
            
        elif "关于" in question_lower or len(question.strip()) < 6:
            return "ambiguous"
            
        elif "不" in question_lower or "没" in question_lower or "无" in question_lower:
            return "negative"
            
        else:
            # 对于无法规则判断的，使用语义相似度
            return self.semantic_classify(question)
    
    def semantic_classify(self, question: str) -> str:
        """基于语义的问题分类"""
        type_examples = {
            "factual": ["需要提供什么材料", "具体要求是什么", "时间限制"],
            "explanatory": ["解释这个概念", "什么是风险管理", "如何理解"],
            "inferential": ["分析原因", "比较差异", "为什么需要"],
            "boundary": ["什么情况下", "例外情况", "特殊条件"],
            "ambiguous": ["材料", "要求", "测试"],
            "negative": ["不需要什么", "不包括哪些", "免除条件"]
        }
        
        question_embedding = self.embedding_model.encode([question])
        best_score = -1
        best_type = "factual"  # 默认类型
        
        for q_type, examples in type_examples.items():
            example_embeddings = self.embedding_model.encode(examples)
            similarities = np.dot(example_embeddings, question_embedding.T)
            max_similarity = np.max(similarities)
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_type = q_type
                
        return best_type

    def retrieve_relevant_paragraphs(self, query: str, question_type: str, num_results: int = 3) -> List[Dict]:
        """根据问题类型使用不同的检索方法"""
        if not self.knowledge_base or self.vector_db is None:
            raise RuntimeError("Knowledge base not initialized")
        
        # 根据问题类型选择检索方法
        retrieval_method = self.retrieval_methods[question_type]
        
        # 根据问题类型调整检索参数
        retrieval_params = self._get_retrieval_params(question_type)
        
        # 调用相应的检索方法
        if retrieval_method == "faiss_exact_search":
            candidates = self._faiss_exact_search(query, retrieval_params['initial_results'])
        elif retrieval_method == "semantic_cluster_search":
            candidates = self._semantic_cluster_search(query, retrieval_params['initial_results'])
        elif retrieval_method == "graph_based_search":
            candidates = self._graph_based_search(query, retrieval_params['initial_results'])
        elif retrieval_method == "hybrid_weighted_search":
            candidates = self._hybrid_weighted_search(query, retrieval_params['initial_results'])
        elif retrieval_method == "diverse_mmr_search":
            candidates = self._diverse_mmr_search(query, retrieval_params['initial_results'])
        elif retrieval_method == "inverted_index_search":
            candidates = self._inverted_index_search(query, retrieval_params['initial_results'])
        else:
            candidates = self._retrieve_by_embedding(query, retrieval_params['initial_results'])
        
        # 应用重排序
        if self.use_reranking and len(candidates) > num_results:
            final_results = self._rerank_results(query, candidates, num_results, question_type)
        else:
            final_results = candidates[:num_results]
        
        return final_results
    
    def _get_retrieval_params(self, question_type: str) -> Dict[str, int]:
        """根据问题类型获取检索参数"""
        params = {
            'factual': {'initial_results': 10, 'reranking_top_k': 8, 'final_results': 3},
            'explanatory': {'initial_results': 12, 'reranking_top_k': 10, 'final_results': 4},
            'inferential': {'initial_results': 15, 'reranking_top_k': 12, 'final_results': 5},
            'boundary': {'initial_results': 10, 'reranking_top_k': 8, 'final_results': 3},
            'ambiguous': {'initial_results': 20, 'reranking_top_k': 15, 'final_results': 6},
            'negative': {'initial_results': 10, 'reranking_top_k': 8, 'final_results': 3}
        }
        return params.get(question_type, {'initial_results': 10, 'reranking_top_k': 8, 'final_results': 4})
    
    def _faiss_exact_search(self, query: str, num_results: int) -> List[Dict]:
        """FAISS精确向量检索 - 用于事实性问题"""
        query_vector = self.embedding_model.encode(
            query, 
            convert_to_tensor=False,
            normalize_embeddings=True
        ).astype('float32')
        
        # FAISS搜索
        distances, indices = self.faiss_index.search(query_vector.reshape(1, -1), num_results)
        
        result_list = []
        for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
            if index < len(self.knowledge_base):
                result_list.append({
                    "content": self.knowledge_base[index]["content"],
                    "similarity": float(distance),
                    "index": int(index)
                })
        
        return result_list
    
    def _semantic_cluster_search(self, query: str, num_results: int) -> List[Dict]:
        """语义聚类检索 - 用于解释性问题"""
        # 使用LSA进行语义空间检索
        query_tfidf = self.tfidf_vectorizer.transform([query])
        query_lsa = self.lsa.transform(query_tfidf)
        query_lsa = Normalizer(copy=False).fit_transform(query_lsa)
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_lsa, self.lsa_matrix).flatten()
        
        # 获取前N个最相似的段落
        top_indices = similarities.argsort()[-num_results:][::-1]
        
        result_list = []
        for index in top_indices:
            result_list.append({
                "content": self.knowledge_base[index]["content"],
                "similarity": float(similarities[index]),
                "index": int(index)
            })
        
        return result_list
    
    def _graph_based_search(self, query: str, num_results: int) -> List[Dict]:
        """基于图的检索 - 用于推理性问题"""
        # 首先找到与查询最相关的节点
        query_vector = self.embedding_model.encode(
            query, 
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        
        # 计算与所有文档的相似度
        document_vectors = self.vector_db.numpy() if hasattr(self.vector_db, 'numpy') else self.vector_db
        similarities = np.dot(document_vectors, query_vector.T).flatten()
        
        # 找到最相关的起始节点
        start_node = np.argmax(similarities)
        
        # 在图中进行广度优先搜索，收集相关节点
        visited = set()
        results = []
        queue = [(start_node, similarities[start_node])]
        
        while queue and len(results) < num_results * 2:
            node, score = queue.pop(0)
            if node not in visited:
                visited.add(node)
                results.append((node, score))
                
                # 添加邻居节点
                for neighbor in self.document_graph.neighbors(node):
                    if neighbor not in visited:
                        edge_weight = self.document_graph[node][neighbor]['weight']
                        neighbor_score = score * 0.7 + similarities[neighbor] * 0.3
                        queue.append((neighbor, neighbor_score))
            
            # 按分数排序队列
            queue.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前N个结果
        results.sort(key=lambda x: x[1], reverse=True)
        result_list = []
        for index, score in results[:num_results]:
            result_list.append({
                "content": self.knowledge_base[index]["content"],
                "similarity": float(score),
                "index": int(index)
            })
        
        return result_list
    
    def _hybrid_weighted_search(self, query: str, num_results: int) -> List[Dict]:
        """混合加权检索 - 用于边界测试问题"""
        # 向量检索
        vector_results = self._retrieve_by_embedding(query, num_results * 2)
        
        # TF-IDF检索
        tfidf_results = self._retrieve_by_tfidf(query, num_results * 2)
        
        # 合并结果并加权
        combined_scores = {}
        for result in vector_results:
            idx = result["index"]
            combined_scores[idx] = combined_scores.get(idx, 0) + result["similarity"] * 0.6
        
        for result in tfidf_results:
            idx = result["index"]
            combined_scores[idx] = combined_scores.get(idx, 0) + result["similarity"] * 0.4
        
        # 选择最高分的结果
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:num_results]
        
        result_list = []
        for index, score in sorted_indices:
            result_list.append({
                "content": self.knowledge_base[index]["content"],
                "similarity": float(score),
                "index": int(index)
            })
        
        return result_list
    
    def _diverse_mmr_search(self, query: str, num_results: int) -> List[Dict]:
        """多样性MMR检索 - 用于模糊查询"""
        # 获取初始候选集
        candidates = self._retrieve_by_embedding(query, num_results * 3)
        
        if len(candidates) <= num_results:
            return candidates
        
        # MMR重排序
        candidate_contents = [candidate["content"] for candidate in candidates]
        candidate_vectors = self.embedding_model.encode(
            candidate_contents, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        similarity_matrix = util.pytorch_cos_sim(candidate_vectors, candidate_vectors)
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
        
        # MMR重排序
        lambda_param = 0.5  # 平衡相关性和多样性
        
        while len(selected_indices) < min(num_results, len(candidates)) and candidate_indices:
            best_score = -1
            best_idx = -1
            
            for idx in candidate_indices:
                relevance_score = query_similarities[idx].item()
                diversity_scores = [similarity_matrix[idx][selected_idx].item() for selected_idx in selected_indices]
                diversity_score = max(diversity_scores) if diversity_scores else 0
                mmr_score = lambda_param * relevance_score - (1 - lambda_param) * diversity_score
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
        
        return [candidates[idx] for idx in selected_indices]
    
    def _inverted_index_search(self, query: str, num_results: int) -> List[Dict]:
        """倒排索引检索 - 用于否定问题"""
        # 使用TF-IDF进行关键词检索，但针对否定问题优化
        query_terms = jieba.cut(query)
        
        # 识别否定词
        negation_terms = ["不", "没", "无", "非", "未", "否"]
        positive_terms = [term for term in query_terms if term not in negation_terms]
        negative_terms = [term for term in query_terms if term in negation_terms]
        
        if not positive_terms:
            return self._retrieve_by_tfidf(query, num_results)
        
        # 搜索包含正向术语但不包含否定术语的文档
        result_scores = {}
        
        for i, content in enumerate([entry['content'] for entry in self.knowledge_base]):
            score = 0
            contains_negation = any(neg_term in content for neg_term in negative_terms)
            
            if contains_negation:
                continue  # 跳过包含否定词的文档
            
            # 计算正向术语的得分
            for term in positive_terms:
                if term in content:
                    # 获取术语的TF-IDF权重
                    term_idx = self.tfidf_vectorizer.vocabulary_.get(term)
                    if term_idx is not None:
                        score += self.tfidf_matrix[i, term_idx]
                    else:
                        score += 0.1  # 小权重给未在词汇表中的术语
            
            if score > 0:
                result_scores[i] = score
        
        # 选择最高分的结果
        sorted_indices = sorted(result_scores.items(), key=lambda x: x[1], reverse=True)[:num_results]
        
        result_list = []
        for index, score in sorted_indices:
            result_list.append({
                "content": self.knowledge_base[index]["content"],
                "similarity": float(score),
                "index": int(index)
            })
        
        return result_list
    
    def _retrieve_by_embedding(self, query: str, num_results: int) -> List[Dict]:
        """使用嵌入向量检索相关段落"""
        # Generate query vector
        query_vector = self.embedding_model.encode(
            query, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        # Calculate similarity
        similarity = util.pytorch_cos_sim(query_vector, self.vector_db)[0]
        similarity_results = torch.topk(similarity, k=min(num_results, len(self.knowledge_base)))
        
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
    
    def _rerank_results(self, query: str, candidates: List[Dict], num_results: int, question_type: str) -> List[Dict]:
        """对候选结果进行重排序，考虑问题类型特性"""
        if len(candidates) <= num_results:
            return candidates
        
        # 根据不同问题类型应用不同的重排序策略
        if question_type == 'factual':
            return self._rerank_factual(query, candidates, num_results)
        elif question_type == 'explanatory':
            return self._rerank_explanatory(query, candidates, num_results)
        elif question_type == 'inferential':
            return self._rerank_inferential(query, candidates, num_results)
        elif question_type == 'ambiguous':
            return self._rerank_ambiguous(query, candidates, num_results)
        else:
            return self._default_rerank(query, candidates, num_results)
    
    def _rerank_factual(self, query: str, candidates: List[Dict], num_results: int) -> List[Dict]:
        """事实性问题重排序：优先选择包含具体条款的内容"""
        scored_candidates = []
        for candidate in candidates:
            score = candidate["similarity"]
            content = candidate["content"]
            
            # 加分项：包含数字、条款编号
            if re.search(r'\d+\.|\([一二三四]\)', content):
                score *= 1.3
            # 加分项：内容较短（通常更精准）
            if len(content) < 200:
                score *= 1.2
            # 加分项：包含问题中的关键词
            if any(keyword in content for keyword in query.split()[:3]):
                score *= 1.1
                
            scored_candidates.append((candidate, score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [candidate for candidate, score in scored_candidates[:num_results]]
    
    def _rerank_explanatory(self, query: str, candidates: List[Dict], num_results: int) -> List[Dict]:
        """解释性问题重排序：优先选择包含定义、概述的内容"""
        explanatory_keywords = ["是指", "包括", "分为", "概念", "定义", "概述", "解释"]
        scored_candidates = []
        
        for candidate in candidates:
            score = candidate["similarity"]
            content = candidate["content"]
            
            # 加分项：包含解释性关键词
            keyword_score = sum(1 for keyword in explanatory_keywords if keyword in content)
            score *= (1 + keyword_score * 0.2)
            
            # 加分项：内容长度适中（100-400字符）
            if 100 <= len(content) <= 400:
                score *= 1.1
                
            scored_candidates.append((candidate, score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [candidate for candidate, score in scored_candidates[:num_results]]
    
    def _rerank_inferential(self, query: str, candidates: List[Dict], num_results: int) -> List[Dict]:
        """推理性问题重排序：优先选择包含分析性内容"""
        reasoning_keywords = ["因为", "因此", "所以", "原因", "目的", "分析", "比较"]
        scored_candidates = []
        
        for candidate in candidates:
            score = candidate["similarity"]
            content = candidate["content"]
            
            # 加分项：包含推理关键词
            keyword_score = sum(1 for keyword in reasoning_keywords if keyword in content)
            score *= (1 + keyword_score * 0.15)
            
            # 加分项：内容较长（提供更多上下文）
            if len(content) > 300:
                score *= 1.1
                
            scored_candidates.append((candidate, score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [candidate for candidate, score in scored_candidates[:num_results]]
    
    def _rerank_ambiguous(self, query: str, candidates: List[Dict], num_results: int) -> List[Dict]:
        """模糊查询重排序：优先选择多样化的内容"""
        # 使用MMR算法保证多样性
        candidate_contents = [candidate["content"] for candidate in candidates]
        candidate_vectors = self.embedding_model.encode(
            candidate_contents, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        similarity_matrix = util.pytorch_cos_sim(candidate_vectors, candidate_vectors)
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
        
        # MMR重排序
        lambda_param = 0.6  # 更注重多样性
        
        while len(selected_indices) < min(num_results, len(candidates)) and candidate_indices:
            best_score = -1
            best_idx = -1
            
            for idx in candidate_indices:
                relevance_score = query_similarities[idx].item()
                diversity_scores = [similarity_matrix[idx][selected_idx].item() for selected_idx in selected_indices]
                diversity_score = max(diversity_scores) if diversity_scores else 0
                mmr_score = lambda_param * relevance_score - (1 - lambda_param) * diversity_score
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
        
        return [candidates[idx] for idx in selected_indices]
    
    def _default_rerank(self, query: str, candidates: List[Dict], num_results: int) -> List[Dict]:
        """默认重排序策略"""
        return sorted(candidates, key=lambda x: x["similarity"], reverse=True)[:num_results]
    
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
    
    def generate_prompt(self, question: str, relevant_paragraphs: list, question_type: str) -> str:
        """Generate a prompt for the LLM based on the question type and retrieved paragraphs"""
        paragraphs_text = "\n\n".join([f"[{i+1}] {p['content']}" for i, p in enumerate(relevant_paragraphs)])
        
        # 根据问题类型定制提示词
        type_instructions = {
            "factual": "请基于以下文档内容，准确、简洁地回答这个事实性问题。只使用文档中明确提供的信息，不要添加额外解释。",
            "explanatory": "请基于以下文档内容，详细解释这个概念或问题。确保解释清晰、完整，包含所有相关方面。",
            "inferential": "请基于以下文档内容，进行逻辑推理和分析。展示推理过程，并得出合理的结论。",
            "boundary": "请基于以下文档内容，分析不同情况下的适用条件和边界。明确指出各种情况的区别和适用范围。",
            "ambiguous": "请基于以下文档内容，全面回答这个模糊查询。提供多个相关方面的信息，确保覆盖所有可能性。",
            "negative": "请基于以下文档内容，准确回答这个否定问题。特别注意文档中关于排除、例外和否定情况的内容。"
        }
        
        instruction = type_instructions.get(question_type, "请基于以下文档内容回答问题。")
        
        prompt = f"""{instruction}

文档内容：
{paragraphs_text}

问题：{question}

请严格按照以下要求回答：
1. 只使用提供的文档内容，不要添加外部知识
2. 如果文档中没有相关信息，请明确说明"根据提供的文档，无法回答这个问题"
3. 保持回答准确、专业
4. 引用相关的文档段落编号来支持你的回答

回答："""
        return prompt
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Main method to answer a question using the RAG system"""
        try:
            # 1. 问题类型分类
            question_type = self.classify_question_type(question)
            print(f"Question type: {self.question_types[question_type]}")
            
            # 2. 检索相关段落
            relevant_paragraphs = self.retrieve_relevant_paragraphs(question, question_type, self.final_top_k)
            
            if not relevant_paragraphs:
                return {
                    "question": question,
                    "answer": "抱歉，在知识库中没有找到相关信息来回答这个问题。",
                    "relevant_paragraphs": [],
                    "question_type": self.question_types[question_type]
                }
            
            # 3. 生成提示词并调用API
            prompt = self.generate_prompt(question, relevant_paragraphs, question_type)
            answer = self.call_deepseek_api(prompt)
            
            return {
                "question": question,
                "answer": answer,
                "relevant_paragraphs": relevant_paragraphs,
                "question_type": self.question_types[question_type]
            }
            
        except Exception as exception:
            return {
                "question": question,
                "answer": f"处理问题时发生错误：{str(exception)}",
                "relevant_paragraphs": [],
                "question_type": "未知"
            }

# 使用示例
if __name__ == "__main__":
    # 配置参数
    KNOWLEDGE_BASE_PATH = "/home/lmy/study/lmy/二级医疗器械/河北省官网资料要求.docx"  # 替换为您的知识库文档路径
    MODEL_PATH = "/home/lmy/study/lmy/Model/model3"  # 替换为您的本地模型路径
    ROBERTA_MODEL_PATH = "/home/lmy/study/lmy/Model/model4_Chinese-RoBERTa-wwm-ext"  # Chinese-RoBERTa-wwm-ext模型路径
    DEEPSEEK_API_KEY = "sk-883d825876464ab6966616a3ae887953"  # 替换为您的DeepSeek API密钥
    
    # 初始化系统
    rag_system = MultiTypeDocumentRAGSystem(
        knowledge_base_path=KNOWLEDGE_BASE_PATH,
        model_path=MODEL_PATH,
        deepseek_api_key=DEEPSEEK_API_KEY,
        roberta_model_path=ROBERTA_MODEL_PATH,
        use_advanced_splitting=True,
        use_reranking=True,
        use_hybrid_search=True
    )
    
    # 测试问题
    test_questions = [
        "什么是风险管理？",
        "为什么需要风险评估？",
        "哪些情况下需要进行安全测试？",
        "不包含在测试范围内的内容有哪些？",
        "测试要求"
    ]
    
    # 回答问题
    for question in test_questions:
        print(f"\n问题：{question}")
        result = rag_system.answer_question(question)
        print(f"问题类型：{result['question_type']}")
        print(f"回答：{result['answer']}")
        print(f"相关段落：{len(result['relevant_paragraphs'])}个")