import gradio as gr
import pandas as pd
import json
import re
import os
import docx
import requests
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import torch

# ======================
# 配置部分
# ======================
# DeepSeek API配置
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-883d825876464ab6966616a3ae887953"  # 请替换为您的实际API Key
KNOWLEDGE_BASE_PATH = "/home/dockeruser/lmy/二级医疗器械/uft82.csv"  # 替换为实际路径

# 嵌入模型配置 - 本地模型路径
LOCAL_MODEL_PATH = "/home/dockeruser/lmy/Model/model"  # 替换为实际的本地模型路径
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动检测GPU

# ======================
# 系统初始化
# ======================
class MedicalDeviceClassifier:
    def __init__(self):
        self.knowledge_base = None
        self.embedding_model = None
        self.embeddings = None
        self.initialize_knowledge_base()

    def initialize_knowledge_base(self):
        """加载并初始化知识库（使用本地向量嵌入模型）"""
        try:
            # 读取知识库CSV文件
            if not os.path.exists(KNOWLEDGE_BASE_PATH):
                raise FileNotFoundError(f"知识库文件不存在: {KNOWLEDGE_BASE_PATH}")
            
            df = pd.read_csv(KNOWLEDGE_BASE_PATH)
            
            # 验证必要字段
            required_columns = ['row_id', 'desc', 'intended_use', 'name', 'grade']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("CSV文件缺少必要列")
            
            self.knowledge_base = df.to_dict('records')
            
            # 从本地路径初始化嵌入模型
            print(f"正在从本地加载嵌入模型({LOCAL_MODEL_PATH})，设备: {EMBEDDING_DEVICE}...")
            if not os.path.exists(LOCAL_MODEL_PATH):
                raise FileNotFoundError(f"本地模型路径不存在: {LOCAL_MODEL_PATH}")
                
            self.embedding_model = SentenceTransformer(
                model_name_or_path=LOCAL_MODEL_PATH,
                device=EMBEDDING_DEVICE
            )
            
            # 为知识库生成嵌入向量
            print("正在生成知识库嵌入向量...")
            corpus = [
                f"{row['name']}。{row['desc']}。用于{row['intended_use']}" 
                for row in self.knowledge_base
            ]
            self.embeddings = self.embedding_model.encode(
                corpus, 
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True,
                batch_size=32  # 根据内存调整批大小
            )
            
            print(f"知识库初始化成功，共加载{len(self.knowledge_base)}条记录")
        except Exception as e:
            print(f"知识库初始化失败: {str(e)}")
            raise

    def call_deepseek_api(self, prompt: str) -> str:
        """调用DeepSeek API进行推理"""
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": """你是一位专业的医疗器械分类专家。请严格遵循以下要求：
1. 必须明确给出医疗器械分类等级（I类/II类/III类）
2. 必须基于提供的知识库条目进行判断
3. 最终回答必须包含以下JSON结构：
```json
{
  "classification": "I/II/III",
  "confidence": 0.0-1.0,
  "rationale": "分类依据说明",
  "matched_id": 匹配的条目ID,
  "missing_info": ["需要补充的字段"]
}
```"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 800
        }
        
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API请求失败: {str(e)}")
        except KeyError:
            raise RuntimeError("API返回格式异常")

    def parse_document(self, file_path: str) -> str:
        """优化版DOCX文档解析
        专注于提取文本内容，合并被错误分段的句子
        """
        try:
            doc = docx.Document(file_path)
            full_text = []
            current_line = ""
            
            for para in doc.paragraphs:
                text = para.text.strip()
                
                if not text:
                    if current_line:  # 遇到空行时提交当前行
                        full_text.append(current_line)
                        current_line = ""
                    continue
                    
                # 判断是否是句子结尾（中文标点）
                if text.endswith(('。', '；', '！', '？', '）', '」', '.', ';', '!', '?')):
                    if current_line:
                        full_text.append(current_line + text)
                        current_line = ""
                    else:
                        full_text.append(text)
                else:
                    current_line += text
            
            # 添加最后未完成的行
            if current_line:
                full_text.append(current_line)
                
            return "\n".join(full_text)
        except Exception as e:
            raise ValueError(f"文档解析失败: {str(e)}")

    def extract_fields(self, text: str) -> dict:
        """精准字段提取
        专注于提取产品描述、预期用途和品名三个关键字段
        """
        # 预处理：标准化文本格式
        text = re.sub(r'[:：]\s*', ': ', text)  # 统一分隔符格式
        
        # 定义多级匹配模式（按优先级排序）
        patterns = {
            "desc": [
                r"产品描述[:：]\s*(.+?)(?=\n|$)",  # 带标签格式
                r"【产品描述】(.+?)(?=【|$)",      # 方括号格式
                r"描述[:：]\s*(.+?)(?=\n|$)",     # 简略标签
                r"(?<=产品概述[:：]).+?(?=\n|$)"  # 后向断言
            ],
            "intended_use": [
                r"预期用途[:：]\s*(.+?)(?=\n|$)",
                r"【预期用途】(.+?)(?=【|$)",
                r"用途[:：]\s*(.+?)(?=\n|$)",
                r"用于(.+?)(?=\n|$)",             # 无标签格式
                r"适用范围[:：]\s*(.+?)(?=\n|$)"
            ],
            "name": [
                r"品名[:：]\s*(.+?)(?=\n|$)",
                r"【产品名称】(.+?)(?=【|$)",
                r"名称[:：]\s*(.+?)(?=\n|$)",
                r"产品名[:：]\s*(.+?)(?=\n|$)",
                r"注册名称[:：]\s*(.+?)(?=\n|$)"
            ]
        }
        
        result = {}
        for field, field_patterns in patterns.items():
            # 尝试所有模式直到匹配成功
            for pattern in field_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    extracted = match.group(1).strip()
                    # 简单清理提取结果
                    extracted = re.sub(r'^[:：\s]+|[:：\s]+$', '', extracted)
                    result[field] = extracted
                    break
            else:
                result[field] = ""
        
        # 智能回退：如果标准模式没找到，尝试顺序提取
        if not all(result.values()):
            lines = [line for line in text.split('\n') if line.strip()]
            if len(lines) >= 3:
                # 智能分配：最长的通常是描述，包含"用于"的是用途，最短的是名称
                lines_sorted = sorted(lines, key=len)
                if not result["desc"]:
                    result["desc"] = lines_sorted[-1]  # 最长的
                if not result["intended_use"] and "用于" in text:
                    for line in lines:
                        if "用于" in line:
                            result["intended_use"] = line
                            break
                if not result["name"]:
                    result["name"] = lines_sorted[0]  # 最短的
        
        # 最终验证和清理
        result = {k: v.strip() for k, v in result.items()}
        
        # 验证字段完整性
        missing = [k for k, v in result.items() if not v]
        if missing:
            raise ValueError(f"文档中缺少以下必要字段: {', '.join(missing)}")
        
        return result

    def retrieve_candidates(self, query: dict, top_k: int = 3) -> List[Dict]:
        """语义检索最相似的候选条目"""
        if not self.knowledge_base or self.embeddings is None:
            raise RuntimeError("知识库未初始化")
        
        # 生成查询嵌入
        query_text = f"{query['name']}。{query['desc']}。用于{query['intended_use']}"
        query_embedding = self.embedding_model.encode(
            query_text, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        # 计算语义相似度（使用余弦相似度）
        cos_scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        
        # 组合结果
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            row = self.knowledge_base[idx]
            results.append({
                "row_id": row["row_id"],
                "desc": row["desc"],
                "intended_use": row["intended_use"],
                "name": row["name"],
                "grade": row["grade"],
                "similarity": float(score)  # 语义相似度分数
            })
        
        return results

    def generate_prompt(self, device_info: dict, candidates: list) -> str:
        """生成给LLM的提示词"""
        candidate_str = "\n".join(
            f"【条目#{c['row_id']}】\n"
            f"描述: {c['desc']}\n"
            f"用途: {c['intended_use']}\n"
            f"名称: {c['name']}\n"
            f"类别: {c['grade']}类\n"
            f"语义相似度: {c['similarity']:.2f}\n"
            for c in candidates
        )
        
        return f"""请对以下医疗器械进行分类分析：

===== 待分类设备 =====
描述: {device_info['desc']}
用途: {device_info['intended_use']}
名称: {device_info['name']}

===== 知识库候选条目 =====
{candidate_str}

请严格按照要求返回JSON格式结果，必须包含：
1. 明确的分类等级（I/II/III）
2. 置信度（0-1）
3. 详细的分类依据
4. 匹配的条目ID（如无可写-1）
5. 需要补充的信息（如无可写空列表）"""

    def process_document(self, file_path: str) -> dict:
        """处理文档的主流程"""
        try:
            # 1. 解析文档
            text = self.parse_document(file_path)
            device_info = self.extract_fields(text)
            
            # 2. 语义检索候选
            candidates = self.retrieve_candidates(device_info)
            
            # 3. 生成Prompt并调用API
            prompt = self.generate_prompt(device_info, candidates)
            api_response = self.call_deepseek_api(prompt)
            
            # 4. 提取JSON结果
            json_match = re.search(r"```json\n(.+?)\n```", api_response, re.DOTALL)
            if not json_match:
                raise ValueError("未找到有效的JSON响应")
            
            result = json.loads(json_match.group(1))
            
            # 验证结果格式
            required_fields = ["classification", "confidence", "rationale"]
            if not all(field in result for field in required_fields):
                raise ValueError("API返回缺少必要字段")
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "classification": "未知",
                "confidence": 0,
                "rationale": "处理过程中出现错误"
            }

def format_output(result: dict) -> str:
    """格式化输出结果"""
    if "error" in result:
        return f"❌ 处理错误: {result['error']}"
    
    # 分类等级映射
    class_map = {
        "I": "Ⅰ类（低风险）",
        "II": "Ⅱ类（中风险）",
        "III": "Ⅲ类（高风险）",
        "unknown": "未知类别"
    }
    
    classification = class_map.get(result["classification"].upper(), class_map["unknown"])
    confidence = f"{result['confidence']*100:.1f}%" if 'confidence' in result else "未知"
    matched_id = result.get("matched_id", "无")
    rationale = result.get("rationale", "无说明")
    
    # 构建Markdown输出
    output = f"""## 🏥 医疗器械分类结果

**🔍 分类等级**: {classification}  
**📊 置信度**: {confidence}  
**🔗 匹配条目ID**: {matched_id}

### 📝 分类依据
{rationale}
"""
    
    # 添加补充信息提示
    if result.get("missing_info"):
        output += f"\n\n⚠️ **需要补充的信息**: {', '.join(result['missing_info'])}"
    
    # 添加免责声明
    output += "\n\n---\n*注：本结果基于AI分析生成，仅供参考。正式分类需以监管部门认定为准。*"
    
    return output

# 初始化分类器
classifier = MedicalDeviceClassifier()

with gr.Blocks(title="医疗器械智能分类系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""<h1 align="center">🏥 医疗器械分类系统（本地模型版）</h1>""")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 上传文档")
            file_input = gr.File(
                label="上传医疗器械文档",
                file_types=[".docx"],
                type="filepath"
            )
            submit_btn = gr.Button("开始分类", variant="primary")
            
            gr.Markdown("### 📝 文档格式要求")
            gr.Markdown("""
            请确保文档包含以下字段（中英文均可）：
            - **产品描述**: 器械的功能和特征
            - **预期用途**: 临床用途和使用方法
            - **品名**: 产品注册名称
            
            示例格式：
            ```
            产品描述: 高频射频手术系统
            预期用途: 用于外科手术中的组织切割和凝血
            品名: 高频电刀
            ```
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 分类结果")
            output = gr.Markdown(
                label="分析结果",
                value="等待分析...",
                show_copy_button=True
            )
    
    # 处理逻辑
    submit_btn.click(
        fn=lambda f: format_output(classifier.process_document(f)),
        inputs=file_input,
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        show_error=True
    )