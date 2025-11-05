import torch
from transformers import BertTokenizer, BertForSequenceClassification
import jieba

class QuestionClassifier:
    def __init__(self, roberta_model_path: str):
        """
        问题分类器 - 单独提取的问题类型分类功能
        
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
            return self.classify_question_type(question), None
    
    def classify_question_type(self, question: str) -> str:
        """基于规则的问题类型分类（备用方法）"""
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
    
    def get_detailed_classification(self, question: str) -> dict:
        """获取详细的问题分类信息"""
        question_type, probabilities = self.classify_question_type_with_roberta(question)
        
        result = {
            'question': question,
            'question_type': question_type,
            'question_type_cn': self.question_types[question_type],
            'probabilities': None
        }
        
        if probabilities is not None:
            result['probabilities'] = {
                self.question_types[self.label_to_type[i]]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        
        return result
    
    def batch_classify(self, questions: list) -> list:
        """批量分类问题"""
        results = []
        for question in questions:
            results.append(self.get_detailed_classification(question))
        return results

# 使用示例
if __name__ == "__main__":
    # 配置参数
    roberta_model_path = "/home/lmy/study/lmy/Model/model4_Chinese-RoBERTa-wwm-ext"
    
    # 初始化分类器
    classifier = QuestionClassifier(roberta_model_path)
    
    # 测试问题集
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
    
    # 单个问题测试
    print("\n--- 单个问题测试 ---")
    test_question = "申报产品需要提供哪些关联文件？"
    result = classifier.get_detailed_classification(test_question)
    print(f"问题: {result['question']}")
    print(f"类型: {result['question_type_cn']}")
    if result['probabilities']:
        print("概率分布:")
        for type_name, prob in result['probabilities'].items():
            print(f"  {type_name}: {prob:.4f}")
    
    # 批量测试
    print("\n--- 批量测试结果 ---")
    batch_results = classifier.batch_classify(test_questions)
    
    for result in batch_results:
        print(f"问题: {result['question'][:20]}...")
        print(f"  类型: {result['question_type_cn']}")
        if result['probabilities']:
            top_type = max(result['probabilities'].items(), key=lambda x: x[1])
            print(f"  最高概率: {top_type[0]} ({top_type[1]:.4f})")
        print()
    
    # 输出详细概率表格
    print("\n--- 详细概率分布 ---")
    print("问题\t\t\t\t\t事实性\t解释性\t推理性\t边界测试\t模糊查询\t否定问题")
    for result in batch_results:
        if result['probabilities']:
            probs = [
                f"{result['probabilities'].get('事实性问题-稠密向量检索', 0):.3f}",
                f"{result['probabilities'].get('解释性问题-图检索', 0):.3f}",
                f"{result['probabilities'].get('推理性问题-层级检索', 0):.3f}",
                f"{result['probabilities'].get('边界测试问题-规则检索', 0):.3f}",
                f"{result['probabilities'].get('模糊查询-多样性检索', 0):.3f}",
                f"{result['probabilities'].get('否定问题-对比检索', 0):.3f}"
            ]
            print(f"{result['question'][:15]}...\t" + "\t".join(probs))