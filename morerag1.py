#事实性问题 - 稠密向量检索
#解释性问题 - 图检索
#推理性问题 - 层级检索
#边界测试问题 - 规则检索
#模糊查询 - 多样性检索
#否定问题 - 对比检索
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
from typing import List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict
import numpy.linalg as LA

class MultiStrategyRAGSystem:
    def __init__(self, knowledge_base_path: str, model_path: str, deepseek_api_key: str, 
                 paragraph_split_length: int = 300, use_advanced_splitting: bool = True):
        """
        多策略RAG系统 - 每种问题类型使用不同的检索方式
        
        Parameters:
            knowledge_base_path: 知识库Word文档路径(.docx)
            model_path: 本地嵌入模型路径  
            deepseek_api_key: DeepSeek API密钥
            paragraph_split_length: 将文档分割成段落的最大长度(字符数)
            use_advanced_splitting: 是否使用高级文档分割
        """
        self.knowledge_base_path = knowledge_base_path
        self.model_path = model_path
        self.api_key = deepseek_api_key
        self.paragraph_split_length = paragraph_split_length
        self.use_advanced_splitting = use_advanced_splitting
        
        self.knowledge_base = None
        self.embedding_model = None
        self.vector_db = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.graph_index = None  # 图检索索引
        self.keyword_index = None  # 关键词索引
        self.semantic_role_index = None  # 语义角色索引
        
        # 问题类型与检索方式映射
        self.retrieval_strategies = {
            "factual": self.fact_dense_retrieval,
            "explanatory": self.explanatory_graph_retrieval,
            "inferential": self.inferential_hierarchical_retrieval,
            "boundary": self.boundary_rule_retrieval,
            "ambiguous": self.ambiguous_diverse_retrieval,
            "negative": self.negative_contrast_retrieval
        }
        
        self.question_types = {
            "factual": "事实性问题-稠密向量检索",
            "explanatory": "解释性问题-图检索",
            "inferential": "推理性问题-层级检索",
            "boundary": "边界测试问题-规则检索",
            "ambiguous": "模糊查询-多样性检索",
            "negative": "否定问题-对比检索"
        }
        
        self.initialize_knowledge_base()
    
    def parse_word_document(self, file_path: str) -> List[Dict]:
        """Parse the Word document and split the content into paragraphs"""
        try:
            doc = Document(file_path)
            all_text = []
            
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
    
    def advanced_document_splitting(self, text_list: List[str]) -> List[Dict]:
        """使用Jieba分词+规则实现高级文档分割"""
        full_text = ' '.join(text_list)
        sentences = []
        current_sentence = []

        jieba.initialize()
        
        for word in jieba.cut(full_text):
            current_sentence.append(word)
            if word in ['。', '！', '？', '；', '…', '，']:
                sentence = ''.join(current_sentence).strip()
                if sentence:
                    sentences.append(sentence)
                current_sentence = []
        
        if current_sentence:
            remaining_sentence = ''.join(current_sentence).strip()
            if remaining_sentence:
                sentences.append(remaining_sentence)
        
        paragraphs_list = []
        current_paragraph = ""
        overlap_text = ""
        
        for i, sentence in enumerate(sentences):
            if len(current_paragraph) + len(sentence) < self.paragraph_split_length:
                current_paragraph += " " + sentence if current_paragraph else sentence
            else:
                if current_paragraph:
                    paragraphs_list.append({"content": current_paragraph})
                current_paragraph = overlap_text + " " + sentence if overlap_text else sentence
                overlap_text = " ".join(sentences[max(0, i - 3):i])
        
        if current_paragraph:
            paragraphs_list.append({"content": current_paragraph})
        
        return paragraphs_list
    
    def simple_document_splitting(self, text_list: List[str]) -> List[Dict]:
        """简单的文档分割"""
        paragraphs_list = []
        current_paragraph = ""
        
        for text in text_list:
            if len(current_paragraph) + len(text) < self.paragraph_split_length:
                current_paragraph += " " + text if current_paragraph else text
            else:
                if current_paragraph:
                    paragraphs_list.append({"content": current_paragraph})
                current_paragraph = text
        
        if current_paragraph:
            paragraphs_list.append({"content": current_paragraph})
        
        return paragraphs_list
    
    def initialize_knowledge_base(self):
        """初始化知识库和所有检索索引"""
        try:
            if not os.path.exists(self.knowledge_base_path):
                raise FileNotFoundError(f"Knowledge base file not found: {self.knowledge_base_path}")
            
            if not self.knowledge_base_path.endswith('.docx'):
                raise ValueError("The knowledge base file must be in .docx format")
            
            self.knowledge_base = self.parse_word_document(self.knowledge_base_path)
            
            print(f"Loading embedding model ({self.model_path})...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
                
            self.embedding_model = SentenceTransformer(
                model_name_or_path=self.model_path,
                device=device
            )
            
            print("Generating vectors for document paragraphs...")
            paragraph_contents = [entry['content'] for entry in self.knowledge_base]
            self.vector_db = self.embedding_model.encode(
                paragraph_contents, 
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            
            # 初始化TF-IDF向量器
            print("Initializing TF-IDF vectorizer...")
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(paragraph_contents)
            
            # 初始化图检索索引
            print("Building graph index...")
            self.graph_index = self.build_graph_index()
            
            # 初始化关键词索引
            print("Building keyword index...")
            self.keyword_index = self.build_keyword_index()
            
            # 初始化语义角色索引
            print("Building semantic role index...")
            self.semantic_role_index = self.build_semantic_role_index()
            
            print(f"Knowledge base initialization completed, loaded {len(self.knowledge_base)} document paragraphs")
            
        except Exception as exception:
            print(f"Initialization failed: {str(exception)}")
            raise
    
    def build_graph_index(self):
        """构建图结构索引用于解释性问题"""
        G = nx.Graph()
        
        # 添加节点
        for i, entry in enumerate(self.knowledge_base):
            G.add_node(i, content=entry['content'])
        
        # 计算段落间的相似度并添加边
        paragraph_contents = [entry['content'] for entry in self.knowledge_base]
        vectors = self.vector_db.cpu().numpy()
        
        # 使用余弦相似度构建边
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                similarity = np.dot(vectors[i], vectors[j]) / (LA.norm(vectors[i]) * LA.norm(vectors[j]))
                if similarity > 0.5:  # 相似度阈值
                    G.add_edge(i, j, weight=similarity)
        
        return G
    
    def build_keyword_index(self):
        """构建关键词索引用于规则检索"""
        keyword_index = defaultdict(list)
        
        # 预定义关键词模式
        keyword_patterns = {
            'condition': ['条件', '要求', '标准', '规范', '规定', '准则'],
            'exception': ['例外', '除外', '豁免', '免除', '不需', '无需'],
            'time': ['时间', '期限', '周期', '有效期', '截止日期', '时限'],
            'material': ['材料', '文件', '资料', '证明', '报告', '证书'],
            'scope': ['范围', '适用', '包含', '不包括', '限于', '适用于']
        }
        
        for i, entry in enumerate(self.knowledge_base):
            content = entry['content']
            for category, keywords in keyword_patterns.items():
                for keyword in keywords:
                    if keyword in content:
                        keyword_index[category].append(i)
        
        return keyword_index
    
    def build_semantic_role_index(self):
        """构建语义角色索引用于层级检索"""
        role_patterns = {
            'definition': ['是指', '定义为', '包括', '包含', '指'],
            'reason': ['因为', '由于', '基于', '考虑到', '鉴于'],
            'result': ['因此', '所以', '导致', '造成', '引起'],
            'process': ['步骤', '流程', '程序', '方法', '过程'],
            'example': ['例如', '比如', '如', '举例', '案例']
        }
        
        semantic_index = defaultdict(list)
        
        for i, entry in enumerate(self.knowledge_base):
            content = entry['content']
            for role, patterns in role_patterns.items():
                for pattern in patterns:
                    if pattern in content:
                        semantic_index[role].append(i)
        
        return semantic_index
    
    # 1. 事实性问题 - 稠密向量检索
    def fact_dense_retrieval(self, query: str, num_results: int = 3) -> List[Dict]:
        """使用稠密向量检索事实信息"""
        query_vector = self.embedding_model.encode(
            query, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        similarity = util.pytorch_cos_sim(query_vector, self.vector_db)[0]
        similarity_results = torch.topk(similarity, k=min(num_results*2, len(self.knowledge_base)))
        
        result_list = []
        for score, index in zip(similarity_results[0], similarity_results[1]):
            entry = self.knowledge_base[index]
            result_list.append({
                "content": entry["content"],
                "similarity": float(score),
                "index": int(index),
                "retrieval_method": "dense_vector"
            })
        
        # 事实性问题优先选择包含数字和条款的内容
        scored_results = []
        for result in result_list:
            bonus = 1.0
            content = result['content']
            
            # 数字加分
            if re.search(r'\d+(?:\.\d+)?', content):
                bonus += 0.2
            
            # 条款格式加分
            if re.search(r'[第（]\d+[条款章]', content):
                bonus += 0.3
            
            result['similarity'] *= bonus
            scored_results.append(result)
        
        scored_results.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_results[:num_results]
    
    # 2. 解释性问题 - 图检索
    def explanatory_graph_retrieval(self, query: str, num_results: int = 3) -> List[Dict]:
        """使用图结构检索解释性内容"""
        query_vector = self.embedding_model.encode([query])
        paragraph_contents = [entry['content'] for entry in self.knowledge_base]
        paragraph_vectors = self.embedding_model.encode(paragraph_contents)
        
        # 找到与查询最相关的起始节点
        similarities = cosine_similarity(query_vector, paragraph_vectors)[0]
        start_node = int(np.argmax(similarities))
        
        # 使用PageRank算法找到重要节点
        try:
            pr_scores = nx.pagerank(self.graph_index, alpha=0.85)
        except:
            pr_scores = {i: 1.0 for i in range(len(self.knowledge_base))}
        
        # 从起始节点开始扩散搜索
        visited = set()
        queue = [start_node]
        relevant_nodes = []
        
        while queue and len(relevant_nodes) < num_results * 2:
            node = queue.pop(0)
            if node in visited:
                continue
            
            visited.add(node)
            relevant_nodes.append(node)
            
            # 添加邻居节点（按权重排序）
            neighbors = list(self.graph_index.neighbors(node))
            neighbors.sort(key=lambda x: self.graph_index[node][x]['weight'], reverse=True)
            queue.extend(neighbors)
        
        # 计算最终得分
        result_list = []
        for node in relevant_nodes[:num_results]:
            content = self.knowledge_base[node]['content']
            similarity_score = similarities[node]
            pagerank_score = pr_scores.get(node, 0)
            
            # 解释性关键词加分
            explanatory_bonus = 0
            for keyword in ['是指', '定义为', '包括', '解释', '概念']:
                if keyword in content:
                    explanatory_bonus += 0.1
            
            final_score = similarity_score * 0.7 + pagerank_score * 0.2 + explanatory_bonus
            
            result_list.append({
                "content": content,
                "similarity": float(final_score),
                "index": node,
                "retrieval_method": "graph_retrieval"
            })
        
        return sorted(result_list, key=lambda x: x["similarity"], reverse=True)[:num_results]
    
    # 3. 推理性问题 - 层级检索
    def inferential_hierarchical_retrieval(self, query: str, num_results: int = 3) -> List[Dict]:
        """使用层级结构检索推理相关内容"""
        # 分析查询中的推理关键词
        reasoning_keywords = ['为什么', '原因', '分析', '比较', '差异', '影响']
        query_keywords = [kw for kw in reasoning_keywords if kw in query]
        
        # 获取不同语义角色的段落
        relevant_indices = set()
        
        for role in ['reason', 'result', 'process']:
            if role in self.semantic_role_index:
                relevant_indices.update(self.semantic_role_index[role])
        
        # 计算与查询的相似度
        query_vector = self.embedding_model.encode([query])
        paragraph_contents = [self.knowledge_base[i]['content'] for i in relevant_indices]
        
        if not paragraph_contents:
            # 如果没有语义角色匹配的段落，使用全文
            paragraph_contents = [entry['content'] for entry in self.knowledge_base]
            relevant_indices = range(len(self.knowledge_base))
        
        paragraph_vectors = self.embedding_model.encode(paragraph_contents)
        similarities = cosine_similarity(query_vector, paragraph_vectors)[0]
        
        # 按相似度排序并选择前N个
        sorted_indices = np.argsort(similarities)[::-1][:num_results]
        
        result_list = []
        for idx in sorted_indices:
            original_idx = list(relevant_indices)[idx]
            score = similarities[idx]
            
            # 推理关键词加分
            content = self.knowledge_base[original_idx]['content']
            keyword_score = sum(1 for kw in reasoning_keywords if kw in content) * 0.1
            
            final_score = score + keyword_score
            
            result_list.append({
                "content": content,
                "similarity": float(final_score),
                "index": original_idx,
                "retrieval_method": "hierarchical_retrieval"
            })
        
        return result_list
    
    # 4. 边界测试问题 - 规则检索
    def boundary_rule_retrieval(self, query: str, num_results: int = 3) -> List[Dict]:
        """使用规则匹配检索边界条件"""
        # 分析查询中的边界关键词
        boundary_keywords = ['条件', '情况', '例外', '边界', '限制', '范围']
        
        # 从关键词索引中获取相关段落
        relevant_indices = set()
        
        for category in ['condition', 'scope', 'exception']:
            if category in self.keyword_index:
                relevant_indices.update(self.keyword_index[category])
        
        if not relevant_indices:
            relevant_indices = range(len(self.knowledge_base))
        
        # 计算TF-IDF相似度
        query_tfidf = self.tfidf_vectorizer.transform([query])
        relevant_matrix = self.tfidf_matrix[list(relevant_indices)]
        similarities = cosine_similarity(query_tfidf, relevant_matrix)[0]
        
        # 边界条件规则匹配
        rule_patterns = [
            r'符合.*条件', r'满足.*要求', r'具备.*资格', 
            r'适用于.*情况', r'不适用于.*情况', r'例外情况'
        ]
        
        scored_indices = []
        for i, idx in enumerate(relevant_indices):
            content = self.knowledge_base[idx]['content']
            score = similarities[i]
            
            # 规则匹配加分
            rule_score = 0
            for pattern in rule_patterns:
                if re.search(pattern, content):
                    rule_score += 0.3
            
            # 边界关键词加分
            keyword_score = sum(1 for kw in boundary_keywords if kw in content) * 0.1
            
            final_score = score + rule_score + keyword_score
            scored_indices.append((idx, final_score))
        
        # 排序并选择前N个
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        result_list = []
        for idx, score in scored_indices[:num_results]:
            result_list.append({
                "content": self.knowledge_base[idx]['content'],
                "similarity": float(score),
                "index": idx,
                "retrieval_method": "rule_retrieval"
            })
        
        return result_list
    
    # 5. 模糊查询 - 多样性检索
    def ambiguous_diverse_retrieval(self, query: str, num_results: int = 3) -> List[Dict]:
        """使用MMR算法实现多样性检索"""
        query_vector = self.embedding_model.encode(
            query, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        # 计算所有段落的相似度
        similarities = util.pytorch_cos_sim(query_vector, self.vector_db)[0]
        
        # 获取候选段落
        candidate_indices = torch.topk(similarities, k=min(num_results*3, len(self.knowledge_base)))[1]
        
        # 获取候选向量
        candidate_vectors = self.vector_db[candidate_indices]
        
        # MMR算法实现多样性
        selected_indices = []
        lambda_param = 0.7  # 平衡相关性和多样性
        
        while len(selected_indices) < num_results and len(candidate_indices) > len(selected_indices):
            best_score = -1
            best_idx = -1
            
            for i in range(len(candidate_indices)):
                if i in selected_indices:
                    continue
                
                # 相关性得分
                relevance = similarities[candidate_indices[i]].item()
                
                # 多样性得分
                if selected_indices:
                    diversity_scores = []
                    for selected in selected_indices:
                        selected_idx = candidate_indices[selected]
                        similarity = util.pytorch_cos_sim(
                            candidate_vectors[i].unsqueeze(0), 
                            candidate_vectors[selected].unsqueeze(0)
                        )[0][0].item()
                        diversity_scores.append(similarity)
                    
                    max_diversity = max(diversity_scores)
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_diversity
                else:
                    mmr_score = lambda_param * relevance
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx != -1:
                selected_indices.append(best_idx)
        
        # 构建结果
        result_list = []
        for idx in selected_indices:
            original_idx = candidate_indices[idx].item()
            result_list.append({
                "content": self.knowledge_base[original_idx]['content'],
                "similarity": float(similarities[original_idx]),
                "index": original_idx,
                "retrieval_method": "diverse_retrieval"
            })
        
        return result_list
    
    # 6. 否定问题 - 对比检索
    def negative_contrast_retrieval(self, query: str, num_results: int = 3) -> List[Dict]:
        """使用对比学习的方式检索否定信息"""
        # 分析否定关键词
        negative_keywords = ['不', '不需要', '不包括', '除外', '免除', '无需', '禁止', '不得']
        
        # 构建正面和负面查询
        positive_query = query
        for kw in negative_keywords:
            positive_query = positive_query.replace(kw, '')
        
        # 检索正面相关段落
        positive_vector = self.embedding_model.encode([positive_query])
        similarities = cosine_similarity(positive_vector, self.vector_db.cpu().numpy())[0]
        
        # 过滤包含否定词的段落
        negative_relevant = []
        for i, entry in enumerate(self.knowledge_base):
            content = entry['content']
            
            # 检查是否包含否定信息
            has_negative = any(kw in content for kw in negative_keywords)
            
            # 计算基础相似度
            base_similarity = similarities[i]
            
            # 给否定信息加权
            negative_bonus = 0.5 if has_negative else 0
            
            # 检查关键词索引中的例外情况
            if 'exception' in self.keyword_index and i in self.keyword_index['exception']:
                negative_bonus += 0.3
            
            final_score = base_similarity + negative_bonus
            
            negative_relevant.append((i, final_score, content))
        
        # 排序并选择
        negative_relevant.sort(key=lambda x: x[1], reverse=True)
        
        result_list = []
        for idx, score, content in negative_relevant[:num_results]:
            result_list.append({
                "content": content,
                "similarity": float(score),
                "index": idx,
                "retrieval_method": "contrast_retrieval"
            })
        
        return result_list
    
    def classify_question_type(self, question: str) -> str:
        """分类问题类型"""
        question_lower = question.lower()
        
        # 规则匹配
        if any(word in question_lower for word in ["什么", "哪些", "何时", "谁", "多少", "是否"]):
            if "为什么" in question_lower or "原因" in question_lower:
                return "inferential"
            elif "不" in question_lower or "没" in question_lower or "无" in question_lower:
                return "negative"
            else:
                return "factual"
                
        elif "为什么" in question_lower or "原因" in question_lower or "如何" in question_lower:
            return "inferential"
            
        elif "什么是" in question_lower or "解释" in question_lower:
            return "explanatory"
            
        elif "哪些情况" in question_lower or "什么时候" in question_lower:
            return "boundary"
            
        elif "关于" in question_lower or len(question.strip()) < 6:
            return "ambiguous"
            
        else:
            return "factual"
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """处理用户问题的完整流程"""
        try:
            # 1. 分类问题类型
            question_type = self.classify_question_type(question)
            question_type_cn = self.question_types[question_type]
            print(f"问题类型识别: {question_type_cn}")
            
            # 2. 根据问题类型选择对应的检索策略
            retrieval_func = self.retrieval_strategies[question_type]
            relevant_paragraphs = retrieval_func(question)
            
            # 3. 生成提示词并获取答案
            prompt = self.generate_prompt(question, relevant_paragraphs, question_type)
            rag_answer = self.call_deepseek_api(prompt)
            
            return {
                'question': question,
                'question_type': question_type,
                'question_type_cn': question_type_cn,
                'retrieval_method': relevant_paragraphs[0]['retrieval_method'] if relevant_paragraphs else 'none',
                'relevant_paragraphs': relevant_paragraphs,
                'answer': rag_answer
            }
            
        except Exception as exception:
            return {
                'question': question,
                'question_type': 'error',
                'question_type_cn': '错误',
                'retrieval_method': 'none',
                'relevant_paragraphs': [],
                'answer': f"处理失败: {str(exception)}"
            }
    
    def generate_prompt(self, question: str, relevant_paragraphs: list, question_type: str) -> str:
        """根据问题类型和检索方式生成定制化的提示词"""
        background_knowledge = "\n".join(
            f"[{i+1}] {entry['content'][:200]}... (检索方式: {entry['retrieval_method']}, 相关度: {entry['similarity']:.2f})" 
            for i, entry in enumerate(relevant_paragraphs)
        )
        
        prompt_templates = {
            "factual": f"""基于稠密向量检索到的相关文档，请直接回答事实性问题：

{background_knowledge}

问题：{question}

请提供准确的事实性答案。""",
            
            "explanatory": f"""基于图检索找到的相关文档节点，请解释概念或说明：

{background_knowledge}

问题：{question}

请用通俗易懂的方式解释相关内容。""",
            
            "inferential": f"""基于层级检索得到的语义相关段落，请进行推理分析：

{background_knowledge}

问题：{question}

请分析原因、比较差异或说明影响。""",
            
            "boundary": f"""基于规则检索找到的边界条件文档，请明确回答：

{background_knowledge}

问题：{question}

请明确说明适用条件、例外情况或边界限制。""",
            
            "ambiguous": f"""基于多样性检索得到的广泛相关文档，请澄清问题：

{background_knowledge}

问题：{question}

请澄清问题意图并提供相关的多方面信息。""",
            
            "negative": f"""基于对比检索找到的否定信息文档，请明确回答：

{background_knowledge}

问题：{question}

请明确说明不需要的内容、排除条件或禁止事项。""",
        }
        
        return prompt_templates.get(question_type, f"""基于检索到的文档回答问题：

{background_knowledge}

问题：{question}

请根据文档内容给出准确答案。""")
    
    def call_deepseek_api(self, prompt: str) -> str:
        """调用DeepSeek API"""
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
    
    def evaluate_qa_effect(self, test_questions: List[str]) -> pd.DataFrame:
        """评估问答效果"""
        result_table = []
        
        for question in test_questions:
            result = self.process_question(question)
            result_table.append({
                "测试问题": result['question'],
                "问题类型": result['question_type_cn'],
                "检索方式": result['retrieval_method'],
                "RAG答案": result['answer'],
                "相关段落数": len(result['relevant_paragraphs'])
            })
        
        return pd.DataFrame(result_table)

# 使用示例
if __name__ == "__main__":
    # 配置参数
    knowledge_base_path = "/home/lmy/study/lmy/二级医疗器械/河北省官网资料要求.docx"
    model_path = "/home/lmy/study/lmy/Model/model3"
    deepseek_key = "sk-883d825876464ab6966616a3ae887953"
    
    # 初始化系统
    rag_system = MultiStrategyRAGSystem(
        knowledge_base_path=knowledge_base_path,
        model_path=model_path,
        deepseek_api_key=deepseek_key,
        paragraph_split_length=300
    )
    
    # 测试问题集
    test_questions = [
        "申报产品需要提供哪些关联文件？",
        "什么是生物学评价？",
        "为什么需要进行稳定性研究？",
        "哪些情况下可以免于临床评价？",
        "关于申请材料",
        "不属于监管信息的内容有哪些？"
    ]
    
    # 测试单个问题
    print("=== 单个问题测试 ===")
    for question in test_questions:
        result = rag_system.process_question(question)
        print(f"\n问题: {result['question']}")
        print(f"类型: {result['question_type_cn']}")
        print(f"检索方式: {result['retrieval_method']}")
        print(f"答案: {result['answer'][:100]}...")
    
    # 批量测试
    print("\n=== 批量测试结果 ===")
    results_df = rag_system.evaluate_qa_effect(test_questions)
    print(results_df[['测试问题', '问题类型', '检索方式', '相关段落数']])
    
    # 保存详细结果
    results_df.to_excel("多策略RAG测试结果.xlsx", index=False)
    print("\n详细结果已保存到 多策略RAG测试结果.xlsx")