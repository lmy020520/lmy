# 最初版本
import pandas as pd
import numpy as np
import torch
import requests
import re
import json
import os
from docx import Document
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict

class DocumentRAGTestingSystem:
    def __init__(self, knowledge_base_path: str, model_path: str, deepseek_api_key: str, paragraph_split_length: int = 300):
        """
        初始化文档RAG测试系统
        
        Parameters:
            knowledge_base_path: 知识库Word文档路径(.docx)
            model_path: 本地嵌入模型路径  
            deepseek_api_key: DeepSeek API密钥
            paragraph_split_length: 将文档分割成段落的最大长度(字符数)

        """
        self.knowledge_base_path = knowledge_base_path
        self.model_path = model_path
        self.api_key = deepseek_api_key
        self.paragraph_split_length = paragraph_split_length
        self.knowledge_base = None
        self.embedding_model = None
        self.vector_db = None
        self.initialize_knowledge_base()
    
    def parse_word_document(self, file_path: str) -> List[Dict]:
        """Parse the Word document and split the content into paragraphs"""
        try:
            doc = Document(file_path)
            paragraphs_list = []
            current_paragraph = ""
            
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                
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
            
            if not paragraphs_list:
                raise ValueError("No usable text content found in the document")
            
            return paragraphs_list
        except Exception as e:
            raise ValueError(f"Failed to parse Word document: {str(e)}")
    
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
            self.vector_db = self.embedding_model.encode(
                paragraph_contents, 
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            
            print(f"Knowledge base initialization completed, loaded {len(self.knowledge_base)} document paragraphs")
            
        except Exception as exception:
            print(f"Initialization failed: {str(exception)}")
            raise
    
    def retrieve_relevant_paragraphs(self, query: str, num_results: int = 3) -> List[Dict]:
        """Retrieve the most relevant document paragraphs"""
        if not self.knowledge_base or self.vector_db is None:
            raise RuntimeError("Knowledge base not initialized")
        
        # Generate query vector
        query_vector = self.embedding_model.encode(
            query, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        # Calculate similarity
        similarity = util.pytorch_cos_sim(query_vector, self.vector_db)[0]
        similarity_results = torch.topk(similarity, k=num_results)
        
        # Format results
        result_list = []
        for score, index in zip(similarity_results[0], similarity_results[1]):
            entry = self.knowledge_base[index]
            result_list.append({
                "content": entry["content"],
                "similarity": float(score)
            })
        
        return result_list
    
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
        
        return f"""基于以下文档内容回答问题：

===== 相关文档段落 =====
{background_knowledge}

===== 待回答问题 =====
{question}

请根据上述文档内容直接给出最准确的答案，不要解释。"""
    
    def evaluate_qa_effect(self, test_questions: List[str], standard_answers: List[str]) -> pd.DataFrame:
        """
        Evaluate the Q&A effectiveness
        
        Parameters:
            test_questions: List of questions to test
            standard_answers: Corresponding standard answers for evaluation
        
        Returns:
            A DataFrame containing the test results, including:
            - Test question
            - Standard answer
            - RAG system answer
            - Answer consistency
            - Retrieved relevant paragraphs
        """
        if len(test_questions) != len(standard_answers):
            raise ValueError("The number of test questions and standard answers must be the same")
        
        result_table = []
        
        for question, standard_answer in zip(test_questions, standard_answers):
            try:
                # 1. Retrieve relevant paragraphs
                relevant_paragraphs = self.retrieve_relevant_paragraphs(question)
                
                # 2. Generate prompt and get RAG answer
                prompt = self.generate_prompt(question, relevant_paragraphs)
                rag_answer = self.call_deepseek_api(prompt)
                
                # 3. Evaluate answer consistency
                answer_consistent = (rag_answer.strip() == standard_answer.strip())
                
                # 4. Record result
                result_table.append({
                    "测试问题": question,
                    "标准答案": standard_answer,
                    "RAG答案": rag_answer,
                    "是否一致": answer_consistent,
                    "相关段落": relevant_paragraphs
                })
                
            except Exception as exception:
                result_table.append({
                    "测试问题": question,
                    "标准答案": standard_answer,
                    "RAG答案": f"错误: {str(exception)}",
                    "是否一致": False,
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
            "监管信息部分需要包含哪些内容？",
            "产品列表应以什么形式呈现？",
            "境内申请人需要提供哪些关联文件？",
            "境外申请人需要提供哪些资格证明文件？",
            "什么情况下需要提供主文档授权信？",
            "申报前与监管机构的沟通记录包括哪些内容？",
            "符合性声明需要声明哪些内容？",
            "综述资料中的'产品描述'部分对无源医疗器械有何要求？",
            "有源医疗器械的产品描述应包括哪些内容？",
            "如何描述产品的型号规格差异？",
            "产品包装说明应包括哪些内容？",
            "产品适用范围描述应包括哪些要素？",
            "预期使用环境应描述哪些内容？",
            "产品上市历史资料应包括哪些内容？",
            "非临床资料中的风险管理资料应包括哪些内容？",
            "医疗器械安全和性能基本原则清单的作用是什么？",
            "产品技术要求应包括哪些内容？",
            "可接受的检验报告形式有哪些？",
            "化学和物理性能研究应包括哪些内容？",
            "电气系统安全性研究应包括哪些内容？",
            "辐射安全研究应包括哪些内容？",
            "软件研究资料应包括哪些内容？",
            "生物学特性研究应包括哪些内容？",
            "生物源材料安全性研究应包括哪些内容？",
            "灭菌研究应包括哪些内容？",
            "什么情况下需要进行动物试验研究？",
            "稳定性研究包括哪些方面？",
            "货架有效期研究需证明什么？",
            "使用稳定性研究需证明什么？",
            "运输稳定性研究需证明什么？",
            "临床评价资料应包括哪些主要内容？",
            "同品种临床评价路径需要提供哪些资料？",
            "临床试验路径需要提供哪些资料？",
            "产品说明书应符合哪些要求？",
            "境外申请人提交说明书有什么特殊要求？",
            "标签样稿应符合哪些要求？",
            "质量管理体系文件中的'生产制造信息'应包括什么？",
            "质量管理体系程序包括哪些高层级程序？",
            "管理职责程序包括哪些内容？",
            "资源管理程序的作用是什么？",
            "产品实现程序包括哪些内容？",
            "设计和开发程序的作用是什么？",
            "采购程序的作用是什么？",
            "生产和服务控制程序包括哪些内容？",
            "监视和测量装置控制程序的作用是什么？",
            "质量管理体系的测量、分析和改进程序的作用是什么？",
            "质量管理体系核查需要提交哪些文件？",
            "有净化要求的生产环境需要提供什么证明？",
            "生产工艺流程图应标明哪些内容？",
            "主要生产设备和检验设备清单应包括哪些内容？"
    ]
    
    standard_answers = [
            "监管信息部分应包括章节目录、申请表、术语和缩写词列表(如适用)、产品列表、关联文件、申报前与监管机构的联系情况和沟通记录、符合性声明。",
            "产品列表应以表格形式列出拟申报产品的型号、规格、结构及组成、附件，以及每个型号规格的标识和描述说明。",
            "境内申请人应提供企业营业执照副本或事业单位法人证书复印件；如适用，还需提供创新医疗器械审查说明、应急审批说明、委托生产的相关文件等。",
            "境外申请人需提供企业资格证明文件、准许上市销售的证明文件(如适用)、中国境内指定代理人的委托书等文件。",
            "如申请人引用主文档信息时，需提供由主文档所有者或其备案代理机构出具的授权信，说明引用情况。",
            "包括监管机构回复的申报前沟通、既往注册申报产品的受理号、会议资料、邮件往来、已明确的问题及解决方案等。",
            "需声明产品符合《医疗器械注册与备案管理办法》和相关法规要求、符合分类规则要求、符合国家标准和行业标准，并保证资料真实性。",
            "需描述工作原理、作用机理(如适用)、结构及组成、原材料、交付状态及灭菌方式、结构示意图等内容。",
            "应包括工作原理、作用机理、结构及组成、主要功能及组成部件功能、产品图示等内容。",
            "应采用对比表或带有说明性文字的图片、图表，描述各种型号规格的结构组成、功能、产品特征和运行模式等。",
            "应包括所有产品组成的包装信息，无菌医疗器械需说明无菌屏障系统信息，有微生物限度要求的需说明保持限度的包装信息。",
            "应包括治疗或诊断功能、医疗过程、目标疾病或病况、预期用途、目标用户、使用次数(一次性/重复使用)、组合使用产品等。",
            "应描述使用地点(如医疗机构、家庭等)和可能影响安全有效性的环境条件(如温度、湿度、压力等)。",
            "应包括各国家地区的上市批准时间、销售情况、不良事件和召回情况、销售数量总结及不良事件/召回比率分析等。",
            "应包括风险分析、风险评价、风险控制、剩余风险可接受性评定、产品风险可接受性综合评价等内容。",
            "用于说明产品符合各项适用要求所采用的方法及证明文件，对不适用的要求需说明理由。",
            "应按照相关规定编制，包括适用的强制性标准、产品性能指标等，对不适用标准需提供说明和验证资料。",
            "可接受申请人出具的自检报告或委托有资质的医疗器械检验机构出具的检验报告。",
            "应包括化学/材料表征、物理/机械性能指标确定依据、燃爆风险研究(如适用)、联合使用研究、量效关系研究等。",
            "应包括电气安全性、机械和环境保护以及电磁兼容性的研究，说明适用标准及开展的研究。",
            "应包括符合的辐射安全标准、辐射类型及安全验证资料、辐射防护措施、验收和维护程序信息等。",
            "应包括软件基本信息、实现过程、核心功能、网络安全、现成软件、人工智能、互操作性等内容。",
            "应包括材料及接触性质描述、物理化学信息、评价策略方法、已有数据评价、生物学试验理由等。",
            "应包括材料获取加工过程、病毒灭活工艺验证、免疫原性降低方法验证等支持安全性的资料。",
            "应包括灭菌工艺和无菌保证水平(生产企业灭菌)、推荐灭菌工艺依据(使用者灭菌)、清洁消毒工艺验证等。",
            "经科学决策需通过动物试验验证产品风险控制措施有效性时，应提供动物试验研究资料。",
            "包括货架有效期研究、使用稳定性研究和运输稳定性研究。",
            "需证明在有效期内，按规定运输贮存条件下产品性能功能满足要求，无菌产品保持无菌状态。",
            "需证明在规定使用期限/次数内，正常使用维护情况下产品性能功能满足要求。",
            "需证明在规定的运输条件下，运输环境不会对产品特性和性能造成不利影响。",
            "应包括产品描述和研发背景、临床评价范围、评价路径(同品种或临床试验)、对比资料或临床试验资料等。",
            "需提供申报产品与同品种医疗器械在适用范围、技术特征等方面的对比资料，以及同品种临床数据的收集评估分析。",
            "需提供临床试验方案、伦理委员会意见、临床试验报告、知情同意书样本及临床试验数据库。",
            "应符合《医疗器械说明书和标签管理规定》和相关法规、规章、规范性文件、强制性标准的要求。",
            "境外申请人应提交产品原文说明书。",
            "应符合《医疗器械说明书和标签管理规定》和相关法规、规章、规范性文件、强制性标准的要求。",
            "应包括产品描述信息(工作原理和工艺说明)和一般生产信息(所有生产地址和重要供应商信息)。",
            "包括质量手册、质量方针、质量目标和文件及记录控制程序等高层级质量管理体系程序。",
            "包括质量方针、策划、职责/权限/沟通和管理评审等形成管理保证文件的程序。",
            "用于为实施和维护质量管理体系形成足够资源(人力资源、基础设施等)供应文件的程序。",
            "包括设计和开发、采购、生产和服务控制、监视和测量装置控制等高层级程序。",
            "用于形成从项目初始至设计转换的整个过程中关于产品设计的系统性和受控的开发过程文件。",
            "用于形成符合已制定的质量和/或产品技术参数的采购产品/服务文件的程序。",
            "包括产品的清洁和污染控制、安装和服务活动、过程确认、标识和可追溯性等问题的程序。",
            "用于形成质量管理体系运行中使用的监视和测量设备已受控并持续符合要求文件的程序。",
            "用于形成如何监视、测量、分析和改进以确保产品和体系符合性并保持有效性的文件。",
            "需提交申请人基本情况表、组织机构图、生产场地平面图、环境检测报告、生产工艺流程图、设备清单等。",
            "需提供有资质的检测机构出具的环境检测报告(附平面布局图)复印件和检测机构资质证明。",
            "应标明主要控制点与项目及主要原材料、采购件的来源及质量控制方法。",
            "应包括进货检验、过程检验、出厂最终检验相关设备，如需净化生产还应包括环境监测设备。"
    ]
    
    # Execute the test
    test_results = tester.evaluate_qa_effect(test_questions, standard_answers)
    
    # Print results
    print("\n测试结果:")
    print(test_results[['测试问题', '标准答案', 'RAG答案', '是否一致']])
    
    # Calculate accuracy rate
    accuracy = test_results['是否一致'].mean()
    print(f"\n答案一致率: {accuracy:.2%}")
    
    # Save detailed results
    test_results.to_excel("文档问答测试结果.xlsx", index=False)
    print("详细结果已保存到 文档问答测试结果.xlsx")
    
    # Output inconsistent cases
    if not test_results['是否一致'].all():
        print("\n不一致案例分析:")
        inconsistent_cases = test_results[~test_results['是否一致']]
        for _, row in inconsistent_cases.iterrows():
            print(f"\n问题: {row['测试问题']}")
            print(f"标准答案: {row['标准答案']}")
            print(f"RAG答案: {row['RAG答案']}")
            print("相关段落:")
            for para in row['相关段落']:
                print(f" - 段落: {para['content']} (相关度:{para['similarity']:.2f})")