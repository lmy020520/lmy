import torch
from transformers import BertTokenizer, BertForSequenceClassification
import jieba
import re
import numpy as np
from collections import defaultdict

class EnhancedQuestionClassifier:
    def __init__(self, roberta_model_path: str):
        """
        增强版问题分类器 - 针对医疗器械领域优化
        
        Parameters:
            roberta_model_path: Chinese-RoBERTa-wwm-ext模型路径
        """
        self.roberta_model_path = roberta_model_path
        self.roberta_tokenizer = None
        self.roberta_model = None
        
        # 问题类型标签映射
        self.label_to_type = {
            0: "factual",      # 事实性
            1: "explanatory",  # 解释性
            2: "inferential",  # 推理性
            3: "boundary",     # 边界测试
            4: "ambiguous",    # 模糊查询
            5: "negative"      # 否定问题
        }
        
        self.question_types = {
            "factual": "事实性问题-稠密向量检索",
            "explanatory": "解释性问题-图检索",
            "inferential": "推理性问题-层级检索",
            "boundary": "边界测试问题-规则检索",
            "ambiguous": "模糊查询-多样性检索",
            "negative": "否定问题-对比检索"
        }
        
        # 医疗器械领域特定关键词
        self.domain_keywords = {
            "factual": ["哪些", "什么", "何种", "包含", "包括", "要求", "文件", "资料", "内容", "信息"],
            "explanatory": ["什么是", "解释", "概念", "定义", "含义", "说明", "介绍", "理解", "定义"],
            "inferential": ["为什么", "原因", "如何", "怎样", "分析", "比较", "差异", "影响", "导致", "推断"],
            "boundary": ["若", "如果", "是否", "情况", "条件", "范围", "限制", "要求", "标准", "资格"],
            "ambiguous": ["如何查找", "哪些部分", "什么内容", "相关", "涉及", "定位", "找到", "查询"],
            "negative": ["不", "没", "无", "非", "未", "禁止", "不得", "不需要", "不包括", "除外", "豁免", "免于"]
        }
        
        # 医疗器械领域特定模式
        self.domain_patterns = {
            "factual": [
                r'.*哪些.*', r'.*什么.*', r'.*何种.*', r'.*包含.*', r'.*包括.*',
                r'.*要求.*', r'.*文件.*', r'.*资料.*', r'.*内容.*', r'.*信息.*'
            ],
            "explanatory": [
                r'什么是.*', r'.*解释.*', r'.*概念.*', r'.*定义.*', r'.*含义.*',
                r'.*说明.*', r'.*介绍.*', r'.*理解.*'
            ],
            "inferential": [
                r'为什么.*', r'.*原因.*', r'如何.*', r'.*怎样.*', r'.*分析.*',
                r'.*比较.*', r'.*差异.*', r'.*影响.*', r'.*导致.*', r'.*推断.*'
            ],
            "boundary": [
                r'若.*', r'如果.*', r'.*是否.*', r'.*情况.*', r'.*条件.*',
                r'.*范围.*', r'.*限制.*', r'.*要求.*', r'.*标准.*', r'.*资格.*'
            ],
            "ambiguous": [
                r'如何查找.*', r'.*哪些部分.*', r'.*什么内容.*', r'.*相关.*',
                r'.*涉及.*', r'.*定位.*', r'.*找到.*', r'.*查询.*'
            ],
            "negative": [
                r'.*不.*', r'.*没.*', r'.*无.*', r'.*非.*', r'.*未.*',
                r'.*禁止.*', r'.*不得.*', r'.*不需要.*', r'.*不包括.*',
                r'.*除外.*', r'.*豁免.*', r'.*免于.*'
            ]
        }
        
        self.initialize_roberta_model()
    
    def initialize_roberta_model(self):
        """初始化Chinese-RoBERTa-wwm-ext模型"""
        try:
            print(f"Loading Chinese-RoBERTa-wwm-ext model from {self.roberta_model_path}...")
            self.roberta_tokenizer = BertTokenizer.from_pretrained(self.roberta_model_path)
            self.roberta_model = BertForSequenceClassification.from_pretrained(
                self.roberta_model_path,
                num_labels=6  # 6种问题类型
            )
            
            # 设置为评估模式
            self.roberta_model.eval()
            
            if torch.cuda.is_available():
                self.roberta_model = self.roberta_model.cuda()
                
            print("Chinese-RoBERTa-wwm-ext model loaded successfully!")
            
        except Exception as e:
            print(f"Failed to load RoBERTa model: {str(e)}")
            raise
    
    def enhanced_classify(self, question: str) -> dict:
        """增强版分类：结合模型预测和规则验证"""
        # 模型预测
        model_type, probabilities = self.classify_question_type_with_roberta(question)
        
        # 规则验证和修正
        rule_type = self.rule_based_classification(question)
        
        # 计算置信度
        confidence = self.calculate_confidence(question, model_type, probabilities)
        
        # 最终决策：如果规则分类与模型不同且置信度较低，使用规则分类
        if rule_type != model_type and confidence < 0.7:
            final_type = rule_type
            used_method = "rule_based"
        else:
            final_type = model_type
            used_method = "model_based"
        
        return {
            'question': question,
            'model_prediction': model_type,
            'rule_prediction': rule_type,
            'final_type': final_type,
            'final_type_cn': self.question_types[final_type],
            'confidence': confidence,
            'used_method': used_method,
            'probabilities': probabilities
        }
    
    def classify_question_type_with_roberta(self, question: str) -> tuple:
        """使用Chinese-RoBERTa-wwm-ext模型分类问题类型"""
        try:
            # 准备输入
            inputs = self.roberta_tokenizer(
                question, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            )
            
            # 移动到GPU（如果可用）
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 前向传播
            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_label = torch.argmax(predictions, dim=1).item()
            
            # 获取预测概率
            probabilities = predictions.cpu().numpy()[0]
            
            # 返回预测类型和概率
            return self.label_to_type.get(predicted_label, "factual"), probabilities
        
        except Exception as e:
            print(f"RoBERTa classification failed: {str(e)}")
            # 失败时回退到规则匹配
            return self.rule_based_classification(question), None
    
    def rule_based_classification(self, question: str) -> str:
        """基于规则的分类（针对医疗器械领域优化）"""
        question_lower = question.lower()
        
        # 否定问题检测（优先级最高）
        negative_keywords = ['不', '没', '无', '非', '未', '禁止', '不得', '不需要', '不包括', '除外', '豁免', '免于']
        if any(keyword in question_lower for keyword in negative_keywords):
            return "negative"
        
        # 边界测试问题检测
        boundary_patterns = ['是否', '若', '如果', '情况', '条件', '范围', '限制']
        if any(pattern in question_lower for pattern in boundary_patterns):
            return "boundary"
        
        # 解释性问题检测
        explanatory_patterns = ['什么是', '解释', '概念', '定义', '含义', '说明']
        if any(pattern in question_lower for pattern in explanatory_patterns):
            return "explanatory"
        
        # 推理性问题检测
        inferential_patterns = ['为什么', '原因', '如何', '怎样', '分析', '比较', '差异', '影响']
        if any(pattern in question_lower for pattern in inferential_patterns):
            return "inferential"
        
        # 模糊查询检测
        ambiguous_patterns = ['如何查找', '哪些部分', '什么内容', '相关', '涉及', '定位']
        if any(pattern in question_lower for pattern in ambiguous_patterns):
            return "ambiguous"
        
        # 事实性问题（默认）
        factual_patterns = ['哪些', '什么', '何种', '包含', '包括', '要求']
        if any(pattern in question_lower for pattern in factual_patterns):
            return "factual"
        
        # 默认返回事实性
        return "factual"
    
    def calculate_confidence(self, question: str, predicted_type: str, probabilities: np.ndarray) -> float:
        """计算分类置信度"""
        if probabilities is None:
            # 规则分类的置信度
            return self.calculate_rule_confidence(question, predicted_type)
        
        # 模型分类的置信度
        base_confidence = max(probabilities)
        
        # 关键词匹配加分
        keyword_bonus = 0
        for keyword in self.domain_keywords.get(predicted_type, []):
            if keyword in question:
                keyword_bonus += 0.1
        
        # 模式匹配加分
        pattern_bonus = 0
        for pattern in self.domain_patterns.get(predicted_type, []):
            if re.search(pattern, question):
                pattern_bonus += 0.15
        
        final_confidence = min(1.0, base_confidence + keyword_bonus + pattern_bonus)
        return final_confidence
    
    def calculate_rule_confidence(self, question: str, predicted_type: str) -> float:
        """计算规则分类的置信度"""
        confidence = 0.6  # 基础置信度
        
        # 关键词匹配加分
        for keyword in self.domain_keywords.get(predicted_type, []):
            if keyword in question:
                confidence += 0.05
        
        # 模式匹配加分
        for pattern in self.domain_patterns.get(predicted_type, []):
            if re.search(pattern, question):
                confidence += 0.08
        
        return min(1.0, confidence)
    
    def get_detailed_classification(self, question: str) -> dict:
        """获取详细的问题分类信息"""
        return self.enhanced_classify(question)
    
    def batch_classify(self, questions: list) -> list:
        """批量分类问题"""
        results = []
        for question in questions:
            results.append(self.get_detailed_classification(question))
        return results
    
    def analyze_classification(self, results: list):
        """分析分类结果"""
        print("\n=== 分类结果分析 ===")
        print("问题\t\t\t\t最终分类\t置信度\t方法")
        print("-" * 80)
        
        for result in results:
            print(f"{result['question'][:20]}...\t{result['final_type_cn'][:10]}\t{result['confidence']:.3f}\t{result['used_method']}")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    roberta_model_path = "/home/lmy/study/lmy/Model/model4_Chinese-RoBERTa-wwm-ext"
    
    # 初始化分类器
    classifier = EnhancedQuestionClassifier(roberta_model_path)
    
    # 测试问题集 - 否定问题
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
    
    print("=== 问题分类测试 ===")
    
    # 批量测试
    batch_results = classifier.batch_classify(test_questions)
    
    # 分析结果
    classifier.analyze_classification(batch_results)
    
    # 输出详细结果
    print("\n=== 详细分类结果 ===")
    for result in batch_results:
        print(f"\n问题: {result['question']}")
        print(f"最终分类: {result['final_type_cn']}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"使用方法: {result['used_method']}")
        print(f"模型预测: {result['model_prediction']}")
        print(f"规则预测: {result['rule_prediction']}")
        
        if result['probabilities'] is not None:
            print("概率分布:")
            for i, prob in enumerate(result['probabilities']):
                type_name = classifier.question_types[classifier.label_to_type[i]]
                print(f"  {type_name}: {prob:.4f}")