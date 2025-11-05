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
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

class DocumentToGraphRAGConverter:
    def __init__(self, document_text: str):
        self.document_text = document_text
        self.nodes = []
        self.edges = []
        self.section_hierarchy = []
        self.current_section = None
        
    def clean_text(self, text: str) -> str:
        """清理文本中的特殊字符和多余空格"""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def extract_sections(self) -> List[Dict[str, Any]]:
        """提取文档中的章节结构"""
        sections = []
        lines = self.document_text.split('\n')
        
        current_section = None
        current_subsection = None
        
        for line in lines:
            line = self.clean_text(line)
            if not line:
                continue
                
            # 检测一级标题
            if re.match(r'^[一二三四五六七八九十]+、', line):
                section_title = re.sub(r'^[一二三四五六七八九十]+、', '', line)
                current_section = {
                    'id': f'section_{len(sections)+1}',
                    'type': 'section',
                    'title': section_title,
                    'content': '',
                    'subsections': []
                }
                sections.append(current_section)
                current_subsection = None
                
            # 检测二级标题
            elif re.match(r'^（[一二三四五六七八九十]+）', line):
                subsection_title = re.sub(r'^（[一二三四五六七八九十]+）', '', line)
                if current_section:
                    subsection = {
                        'id': f'subsection_{len(current_section["subsections"])+1}',
                        'type': 'subsection',
                        'title': subsection_title,
                        'content': '',
                        'items': []
                    }
                    current_section['subsections'].append(subsection)
                    current_subsection = subsection
                    
            # 检测三级项目
            elif re.match(r'^\d+\.', line):
                item_title = re.sub(r'^\d+\.', '', line).strip()
                if current_subsection:
                    item = {
                        'id': f'item_{len(current_subsection["items"])+1}',
                        'type': 'item',
                        'title': item_title,
                        'content': ''
                    }
                    current_subsection['items'].append(item)
                    
            # 添加内容到当前层级
            else:
                if current_subsection and current_subsection['items']:
                    current_subsection['items'][-1]['content'] += ' ' + line
                elif current_subsection:
                    current_subsection['content'] += ' ' + line
                elif current_section:
                    current_section['content'] += ' ' + line
                    
        return sections
    
    def extract_entities(self, sections: List[Dict[str, Any]]) -> None:
        """从章节中提取实体和关系"""
        for section in sections:
            # 添加章节节点
            section_node = {
                'id': section['id'],
                'type': 'Section',
                'title': section['title'],
                'content': self.clean_text(section['content'])[:500]
            }
            self.nodes.append(section_node)
            
            # 处理子章节
            for subsection in section['subsections']:
                subsection_node = {
                    'id': subsection['id'],
                    'type': 'Subsection',
                    'title': subsection['title'],
                    'content': self.clean_text(subsection['content'])[:500]
                }
                self.nodes.append(subsection_node)
                
                # 添加章节-子章节关系
                self.edges.append({
                    'source': section['id'],
                    'target': subsection['id'],
                    'relation': 'has_subsection'
                })
                
                # 处理子章节中的项目
                for item in subsection['items']:
                    item_node = {
                        'id': item['id'],
                        'type': 'Requirement' if '应当' in item['title'] or '必须' in item['title'] else 'Guideline',
                        'title': item['title'],
                        'content': self.clean_text(item['content'])[:500]
                    }
                    self.nodes.append(item_node)
                    
                    # 添加子章节-项目关系
                    self.edges.append({
                        'source': subsection['id'],
                        'target': item['id'],
                        'relation': 'contains_item'
                    })
                    
                    # 从项目内容中提取更多实体和关系
                    self.extract_entities_from_content(item['content'], item['id'])
    
    def extract_entities_from_content(self, content: str, parent_id: str) -> None:
        """从内容文本中提取更多实体和关系"""
        # 提取法规引用
        regulations = re.findall(r'《([^》]+)》', content)
        for reg in set(regulations):
            reg_id = f'regulation_{reg}'
            
            if not any(node['id'] == reg_id for node in self.nodes):
                self.nodes.append({
                    'id': reg_id,
                    'type': 'Regulation',
                    'title': reg,
                    'content': ''
                })
            
            self.edges.append({
                'source': parent_id,
                'target': reg_id,
                'relation': 'references'
            })
        
        # 提取文件类型要求
        file_types = re.findall(r'应当提供(.+?(?:文件|资料|证明))', content)
        for ft in set(file_types):
            ft_id = f'filetype_{hash(ft)}'
            
            if not any(node['id'] == ft_id for node in self.nodes):
                self.nodes.append({
                    'id': ft_id,
                    'type': 'RequiredDocument',
                    'title': self.clean_text(ft),
                    'content': ''
                })
                
            self.edges.append({
                'source': parent_id,
                'target': ft_id,
                'relation': 'requires'
            })
    
    def convert_to_graphrag_format(self) -> Dict[str, Any]:
        """转换为GraphRAG可用的格式"""
        sections = self.extract_sections()
        self.extract_entities(sections)
        
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'metadata': {
                'source': '河北省官网医疗器械注册资料',
                'document_type': '医疗器械注册要求'
            }
        }

class GraphEnhancedRAGSystem:
    def __init__(self, knowledge_base_path: str, model_path: str, deepseek_api_key: str, 
                 paragraph_split_length: int = 300, use_advanced_splitting: bool = True,
                 use_reranking: bool = True, use_hybrid_search: bool = True,
                 use_graph_enhancement: bool = True):
        """
        初始化增强版RAG系统
        
        Parameters:
            knowledge_base_path: 知识库Word文档路径(.docx)
            model_path: 本地嵌入模型路径  
            deepseek_api_key: DeepSeek API密钥
            paragraph_split_length: 段落最大长度
            use_advanced_splitting: 是否使用高级文档分割
            use_reranking: 是否使用重排序  
            use_hybrid_search: 是否使用混合搜索
            use_graph_enhancement: 是否使用知识图谱增强
        """
        self.knowledge_base_path = knowledge_base_path
        self.model_path = model_path
        self.api_key = deepseek_api_key
        self.paragraph_split_length = paragraph_split_length
        self.use_advanced_splitting = use_advanced_splitting
        self.use_reranking = use_reranking
        self.use_hybrid_search = use_hybrid_search
        self.use_graph_enhancement = use_graph_enhancement
        
        self.knowledge_base = None
        self.embedding_model = None
        self.vector_db = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.knowledge_graph = None
        
        self.overlap_size = 100
        self.reranking_top_k = 10
        self.final_top_k = 3
        
        self.initialize_system()
    
    def initialize_system(self):
        """初始化整个系统"""
        self.initialize_knowledge_base()
        
        # 如果启用图谱增强，构建知识图谱
        if self.use_graph_enhancement:
            self.initialize_knowledge_graph()
    
    def parse_word_document(self, file_path: str) -> List[Dict]:
        """解析Word文档并分割内容"""
        try:
            doc = Document(file_path)
            all_text = []
            
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    all_text.append(text)
            
            if not all_text:
                raise ValueError("文档中没有可用文本内容")
            
            if self.use_advanced_splitting:
                return self.advanced_document_splitting(all_text)
            else:
                return self.simple_document_splitting(all_text)
            
        except Exception as e:
            raise ValueError(f"解析Word文档失败: {str(e)}")
    
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
    
    def advanced_document_splitting(self, text_list: List[str]) -> List[Dict]:
        """高级文档分割"""
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
    
    def initialize_knowledge_base(self):
        """初始化知识库"""
        try:
            if not os.path.exists(self.knowledge_base_path):
                raise FileNotFoundError(f"知识库文件未找到: {self.knowledge_base_path}")
            
            if not self.knowledge_base_path.endswith('.docx'):
                raise ValueError("知识库文件必须是.docx格式")
            
            self.knowledge_base = self.parse_word_document(self.knowledge_base_path)
            
            print(f"加载嵌入模型 ({self.model_path})...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
                
            self.embedding_model = SentenceTransformer(
                model_name_or_path=self.model_path,
                device=device
            )
            
            print("为文档段落生成向量...")
            paragraph_contents = [entry['content'] for entry in self.knowledge_base]
            self.vector_db = self.embedding_model.encode(
                paragraph_contents, 
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            
            if self.use_hybrid_search:
                print("初始化TF-IDF向量器...")
                self.tfidf_vectorizer = TfidfVectorizer()
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(paragraph_contents)
            
            print(f"知识库初始化完成，加载了 {len(self.knowledge_base)} 个文档段落")
            
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            raise
    
    def initialize_knowledge_graph(self):
        """初始化知识图谱"""
        try:
            print("构建知识图谱...")
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
            
            converter = DocumentToGraphRAGConverter(document_text)
            self.knowledge_graph = converter.convert_to_graphrag_format()
            
            # 为图谱节点生成嵌入向量
            graph_node_texts = [node['title'] + ' ' + node.get('content', '') for node in self.knowledge_graph['nodes']]
            self.graph_node_embeddings = self.embedding_model.encode(
                graph_node_texts,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            
            print(f"知识图谱构建完成，包含 {len(self.knowledge_graph['nodes'])} 个节点和 {len(self.knowledge_graph['edges'])} 条边")
            
        except Exception as e:
            print(f"知识图谱构建失败: {str(e)}")
            self.knowledge_graph = None
    
    def retrieve_relevant_paragraphs(self, query: str, num_results: int = 3) -> List[Dict]:
        """检索相关段落"""
        if not self.knowledge_base or self.vector_db is None:
            raise RuntimeError("知识库未初始化")
        
        if self.use_hybrid_search:
            vector_results = self._retrieve_by_embedding(query, self.reranking_top_k * 2)
            tfidf_results = self._retrieve_by_tfidf(query, self.reranking_top_k * 2)
            
            combined_results = {}
            for result in vector_results + tfidf_results:
                content = result["content"]
                if content not in combined_results:
                    combined_results[content] = result
                else:
                    combined_results[content]["similarity"] = max(
                        combined_results[content]["similarity"], 
                        result["similarity"]
                    )
            
            candidates = sorted(combined_results.values(), key=lambda x: x["similarity"], reverse=True)
            candidates = candidates[:self.reranking_top_k]
        else:
            candidates = self._retrieve_by_embedding(query, self.reranking_top_k)
        
        # 如果启用图谱增强，添加相关知识图谱信息
        if self.use_graph_enhancement and self.knowledge_graph:
            graph_context = self._retrieve_graph_context(query)
            if graph_context:
                candidates.extend(graph_context)
        
        if self.use_reranking and len(candidates) > num_results:
            final_results = self._rerank_results(query, candidates, num_results)
        else:
            final_results = candidates[:num_results]
        
        return final_results
    
    def _retrieve_graph_context(self, query: str) -> List[Dict]:
        """从知识图谱中检索相关上下文"""
        query_vector = self.embedding_model.encode(
            query, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        similarities = util.pytorch_cos_sim(query_vector, self.graph_node_embeddings)[0]
        top_indices = torch.topk(similarities, k=min(5, len(self.knowledge_graph['nodes']))[1])
        
        graph_context = []
        for idx in top_indices:
            node = self.knowledge_graph['nodes'][idx]
            related_nodes = self._get_related_nodes(node['id'])
            
            context_entry = {
                "content": f"{node['title']}: {node.get('content', '')}",
                "similarity": float(similarities[idx]),
                "index": f"graph_{idx}",
                "type": "graph_node",
                "related_info": related_nodes
            }
            graph_context.append(context_entry)
        
        return graph_context
    
    def _get_related_nodes(self, node_id: str) -> List[str]:
        """获取与指定节点相关的节点信息"""
        related_info = []
        
        # 查找所有与该节点相连的边
        for edge in self.knowledge_graph['edges']:
            if edge['source'] == node_id:
                target_node = next((n for n in self.knowledge_graph['nodes'] if n['id'] == edge['target']), None)
                if target_node:
                    related_info.append(f"{edge['relation']}: {target_node['title']}")
            elif edge['target'] == node_id:
                source_node = next((n for n in self.knowledge_graph['nodes'] if n['id'] == edge['source']), None)
                if source_node:
                    related_info.append(f"{edge['relation']} by: {source_node['title']}")
        
        return related_info
    
    def _retrieve_by_embedding(self, query: str, num_results: int) -> List[Dict]:
        """使用嵌入向量检索"""
        query_vector = self.embedding_model.encode(
            query, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        similarity = util.pytorch_cos_sim(query_vector, self.vector_db)[0]
        similarity_results = torch.topk(similarity, k=min(num_results, len(self.knowledge_base)))
        
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
        """使用TF-IDF检索"""
        query_tfidf = self.tfidf_vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        
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
        """重排序结果"""
        if len(candidates) <= num_results:
            return candidates
        
        candidate_contents = [candidate["content"] for candidate in candidates]
        candidate_vectors = self.embedding_model.encode(
            candidate_contents, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        similarity_matrix = util.pytorch_cos_sim(candidate_vectors, candidate_vectors)
        
        selected_indices = []
        candidate_indices = list(range(len(candidates)))
        
        query_vector = self.embedding_model.encode(
            query, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        query_similarities = util.pytorch_cos_sim(query_vector, candidate_vectors)[0]
        most_relevant_idx = torch.argmax(query_similarities).item()
        
        selected_indices.append(most_relevant_idx)
        candidate_indices.remove(most_relevant_idx)
        
        lambda_param = 0.7
        
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
        except Exception as e:
            raise RuntimeError(f"API请求失败: {str(e)}")
    
    def generate_prompt(self, question: str, relevant_paragraphs: list) -> str:
        """生成提示词"""
        background_knowledge = "\n".join(
            f"段落内容: {entry['content']}\n(相关度:{entry['similarity']:.2f})" 
            for entry in relevant_paragraphs
        )
        
        # 提取知识图谱中的相关信息
        graph_knowledge = ""
        if self.use_graph_enhancement:
            graph_entries = [entry for entry in relevant_paragraphs if entry.get('type') == 'graph_node']
            if graph_entries:
                graph_knowledge = "\n===== 相关知识图谱 =====\n"
                for entry in graph_entries:
                    graph_knowledge += f"概念: {entry['content']}\n"
                    if entry.get('related_info'):
                        graph_knowledge += "相关概念:\n- " + "\n- ".join(entry['related_info']) + "\n"
        
        return f"""基于以下文档内容回答问题：

===== 相关文档段落 =====
{background_knowledge}
{graph_knowledge}

===== 待回答问题 =====
{question}

请根据上述内容直接给出最准确的答案，不要解释。"""
    
    def evaluate_qa_effect(self, test_questions: List[str]) -> pd.DataFrame:
        """评估QA效果"""
        result_table = []
        
        for question in test_questions:
            try:
                relevant_paragraphs = self.retrieve_relevant_paragraphs(question)
                prompt = self.generate_prompt(question, relevant_paragraphs)
                rag_answer = self.call_deepseek_api(prompt)
                
                result_table.append({
                    "测试问题": question,
                    "RAG答案": rag_answer,
                    "相关段落": relevant_paragraphs
                })
                
            except Exception as e:
                result_table.append({
                    "测试问题": question,
                    "RAG答案": f"错误: {str(e)}",
                    "相关段落": []
                })
        
        return pd.DataFrame(result_table)

# 示例使用
if __name__ == "__main__":
    # 配置参数
    knowledge_base_path = "/home/dockeruser/lmy/二级医疗器械/河北省官网资料要求.docx"
    model_path = "/home/dockeruser/lmy/Model/model3"
    deepseek_key = "sk-883d825876464ab6966616a3ae887953"
    
    # 初始化测试系统
    tester = GraphEnhancedRAGSystem(
        knowledge_base_path=knowledge_base_path,
        model_path=model_path,
        deepseek_api_key=deepseek_key,
        paragraph_split_length=300,
        use_graph_enhancement=True  # 启用知识图谱增强
    )
    
    # 准备测试问题
    test_questions = [
        "申报产品需要提供哪些关联文件（境内申请人）？",
        "产品技术要求应包括哪些内容？",
        "非临床资料中'生物学特性研究'需包含哪些信息？",
        "境外申请人需提交哪些企业资格证明文件？",
        "医疗器械说明书应符合哪些法规要求？",
        "货架有效期研究需证明什么内容？",
        "软件研究资料需包含哪些基本信息？",
        "动物试验研究的目的是什么？",
        "质量管理体系文件中'生产制造信息'需提供什么？",
        "临床评价路径有哪两种？"
    ]
    
    # 运行测试
    test_results = tester.evaluate_qa_effect(test_questions)
    
    # 打印结果
    print("\n测试结果:")
    print(test_results[['测试问题', 'RAG答案']])
    
    # 保存详细结果
    test_results.to_excel("测试结果_图谱增强版.xlsx", index=False)
    print("详细结果已保存到 测试结果_图谱增强版.xlsx")
    
    # 输出每个问题的相关段落
    for _, row in test_results.iterrows():
        print(f"\n问题: {row['测试问题']}")
        print(f"RAG答案: {row['RAG答案']}")
        print("相关段落:")
        for para in row['相关段落']:
            print(f" - 段落: {para['content']} (相关度:{para['similarity']:.2f})")