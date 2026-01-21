import pandas as pd
import numpy as np
import logging
import jieba
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional

# 创建日志目录（如果不存在）
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler(os.path.join(log_dir, 'data_matcher.log'), encoding='utf-8')  # 输出到文件
    ]
)

logging.info(f"日志目录: {log_dir}")

class DataMatcher:
    """自动化数据匹配与处理程序"""
    
    def __init__(self, file_a: str = 'A.xlsx', file_b: str = 'B.xlsx'):
        """初始化数据匹配器"""
        self.file_a = file_a
        self.file_b = file_b
        self.df_a = None  # A表数据
        self.df_b = None  # B表数据
        self.result_df = None  # 匹配结果
        self.vectorizer = None  # TF-IDF向量化器
        
        # 配置参数
        self.unit_col = '单位'  # 需要处理的单位列名
        self.project_col = '项目名'  # 项目名列名
        self.threshold = 0.5  # 匹配阈值
    
    def load_and_clean_data(self) -> bool:
        """加载并清洗数据"""
        try:
            # 读取数据
            logging.info(f"加载数据文件: {self.file_a}")
            self.df_a = pd.read_excel(self.file_a)
            logging.info(f"加载数据文件: {self.file_b}")
            self.df_b = pd.read_excel(self.file_b)
            
            # 数据清洗
            logging.info("执行数据清洗...")
            
            # 处理缺失值
            if self.df_a.isnull().any().any():
                logging.warning(f"A表存在缺失值: {self.df_a.isnull().sum().to_dict()}")
                self.df_a = self.df_a.dropna()
            
            if self.df_b.isnull().any().any():
                logging.warning(f"B表存在缺失值: {self.df_b.isnull().sum().to_dict()}")
                self.df_b = self.df_b.dropna()
            
            # 去重
            initial_count_a = len(self.df_a)
            self.df_a = self.df_a.drop_duplicates()
            if len(self.df_a) < initial_count_a:
                logging.info(f"A表去重后，数据从{initial_count_a}条减少到{len(self.df_a)}条")
            
            initial_count_b = len(self.df_b)
            self.df_b = self.df_b.drop_duplicates()
            if len(self.df_b) < initial_count_b:
                logging.info(f"B表去重后，数据从{initial_count_b}条减少到{len(self.df_b)}条")
            
            logging.info(f"数据加载完成，A表: {len(self.df_a)}条，B表: {len(self.df_b)}条")
            return True
        except Exception as e:
            logging.error(f"数据加载失败: {e}")
            return False
    
    def group_by_unit(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """按单位分组数据"""
        try:
            # 按单位分组
            groups_a = {unit: group for unit, group in self.df_a.groupby(self.unit_col)}
            groups_b = {unit: group for unit, group in self.df_b.groupby(self.unit_col)}
            
            logging.info(f"按单位分组完成，A表: {len(groups_a)}个单位，B表: {len(groups_b)}个单位")
            
            # 识别在B表中不存在的单位
            missing_units = [unit for unit in groups_a.keys() if unit not in groups_b]
            if missing_units:
                logging.info(f"B表中不存在的单位: {sorted(missing_units)}")
            
            return groups_a, groups_b
        except Exception as e:
            logging.error(f"单位分组失败: {e}")
            return {}, {}
    
    def preprocess_text(self, text: str) -> str:
        """中文文本预处理"""
        if not isinstance(text, str):
            return ""
        
        # 分词
        words = jieba.cut(text)
        # 移除停用词（简单实现）
        # 注意：保留"一期"、"二期"等项目阶段词汇，确保不同阶段项目被识别为不同项目
        stopwords = {'的', '了', '和', '与', '等', '项目', '公司'}
        filtered_words = [word for word in words if word not in stopwords and word.strip()]
        # 重新组合为字符串
        return " ".join(filtered_words)
    
    @staticmethod
    def edit_distance(text1: str, text2: str) -> int:
        """计算编辑距离（Levenshtein距离）"""
        m, n = len(text1), len(text2)
        # 创建二维数组
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化第一行和第一列
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # 填充数组
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,    # 删除
                        dp[i][j - 1] + 1,    # 插入
                        dp[i - 1][j - 1] + 1  # 替换
                    )
        
        return dp[m][n]
    
    def calculate_similarity(self, text1: str, text2: str, method: str = 'cosine') -> float:
        """计算两个文本的相似度"""
        if not text1 or not text2:
            return 0.0
        
        if method == 'cosine':
            # 余弦相似度
            try:
                vecs = self.vectorizer.transform([text1, text2])
                return cosine_similarity(vecs[0], vecs[1])[0][0]
            except Exception as e:
                logging.error(f"余弦相似度计算失败: {e}")
                return 0.0
        elif method == 'levenshtein':
            # 编辑距离相似度（归一化）
            max_len = max(len(text1), len(text2))
            if max_len == 0:
                return 1.0
            distance = self.edit_distance(text1, text2)
            return 1 - distance / max_len
        else:
            logging.error(f"不支持的相似度计算方法: {method}")
            return 0.0
    
    def match_projects(self, group_a: pd.DataFrame, group_b: pd.DataFrame) -> pd.DataFrame:
        """匹配同一单位下的项目"""
        # 提取项目名称
        projects_a = group_a[self.project_col].tolist()
        projects_b = group_b[self.project_col].tolist()
        
        # 预处理所有项目名称
        processed_a = [self.preprocess_text(p) for p in projects_a]
        processed_b = [self.preprocess_text(p) for p in projects_b]
        
        # 训练TF-IDF向量化器
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(processed_a + processed_b)
        
        # 计算相似度矩阵
        similarities = []
        for i, proj_a in enumerate(processed_a):
            best_match = ""
            best_score = 0.0
            
            for j, proj_b in enumerate(processed_b):
                # 计算余弦相似度和编辑距离相似度的加权平均
                cosine_sim = self.calculate_similarity(proj_a, proj_b, 'cosine')
                levenshtein_sim = self.calculate_similarity(proj_a, proj_b, 'levenshtein')
                score = (cosine_sim + levenshtein_sim) / 2
                
                if score > best_score:
                    best_score = score
                    best_match = projects_b[j]
            
            similarities.append((best_match, best_score))
        
        # 生成匹配结果
        result = group_a.copy()
        result['匹配项目名'] = [match for match, _ in similarities]
        result['相似度'] = [round(score, 4) for _, score in similarities]
        result['匹配状态'] = result['相似度'].apply(
            lambda x: "匹配成功" if x > self.threshold else "匹配不成功"
        )
        
        return result
    
    def process_all_units(self) -> bool:
        """处理所有单位"""
        try:
            groups_a, groups_b = self.group_by_unit()
            
            all_results = []
            missing_unit_results = []
            
            # 处理在B表中存在的单位
            common_units = [unit for unit in groups_a.keys() if unit in groups_b.keys()]
            logging.info(f"开始处理{len(common_units)}个共同单位...")
            
            for i, unit in enumerate(common_units, 1):
                logging.info(f"处理单位 {i}/{len(common_units)}: {unit}")
                group_a = groups_a[unit]
                group_b = groups_b[unit]
                
                # 匹配项目
                matched = self.match_projects(group_a, group_b)
                all_results.append(matched)
            
            # 处理在B表中不存在的单位
            missing_units = [unit for unit in groups_a.keys() if unit not in groups_b.keys()]
            if missing_units:
                logging.info(f"处理{len(missing_units)}个缺失单位...")
                for unit in missing_units:
                    group_a = groups_a[unit].copy()
                    group_a['匹配项目名'] = ""
                    group_a['相似度'] = 0.0
                    group_a['匹配状态'] = "无相关单位数据"
                    missing_unit_results.append(group_a)
            
            # 合并结果
            if all_results:
                self.result_df = pd.concat(all_results, ignore_index=True)
            if missing_unit_results:
                missing_df = pd.concat(missing_unit_results, ignore_index=True)
                self.result_df = pd.concat([self.result_df, missing_df], ignore_index=True) if self.result_df is not None else missing_df
            
            logging.info(f"所有单位处理完成，总匹配结果: {len(self.result_df)}条")
            return True
        except Exception as e:
            logging.error(f"处理所有单位失败: {e}")
            return False
    
    def generate_statistics(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """生成统计数据"""
        if self.result_df is None:
            return pd.DataFrame(), pd.DataFrame()
        
        try:
            # 总体统计
            total_count = len(self.result_df)
            match_success = len(self.result_df[self.result_df['匹配状态'] == '匹配成功'])
            match_failed = len(self.result_df[self.result_df['匹配状态'] == '匹配不成功'])
            no_data = len(self.result_df[self.result_df['匹配状态'] == '无相关单位数据'])
            
            overall_stats = pd.DataFrame({
                '统计项': ['总记录数', '匹配成功', '匹配不成功', '无相关单位数据', '匹配成功率'],
                '数值': [
                    total_count,
                    match_success,
                    match_failed,
                    no_data,
                    f"{match_success / total_count * 100:.2f}%"
                ]
            })
            
            # 单位统计
            unit_stats = self.result_df.groupby([self.unit_col, '匹配状态']).size().unstack(fill_value=0)
            unit_stats['总记录数'] = unit_stats.sum(axis=1)
            unit_stats['匹配成功率'] = unit_stats.apply(
                lambda x: f"{x.get('匹配成功', 0) / x['总记录数'] * 100:.2f}%" if x['总记录数'] > 0 else "0.00%",
                axis=1
            )
            
            return overall_stats, unit_stats
        except Exception as e:
            logging.error(f"生成统计数据失败: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def export_results(self, output_file: str = '匹配结果.xlsx') -> bool:
        """导出多工作表结果"""
        try:
            if self.result_df is None:
                logging.error("没有匹配结果可导出")
                return False
            
            # 生成统计数据
            overall_stats, unit_stats = self.generate_statistics()
            
            # 输出统计信息到日志
            if not overall_stats.empty:
                logging.info("=== 数据匹配统计结果 ===")
                for _, row in overall_stats.iterrows():
                    logging.info(f"{row['统计项']}: {row['数值']}")
            
            # 导出到Excel
            logging.info(f"导出结果到文件: {output_file}")
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Sheet1: 完整匹配结果
                self.result_df.to_excel(writer, sheet_name='匹配结果', index=False)
                logging.info("导出Sheet1: 匹配结果")
                
                # Sheet2: 总体统计
                overall_stats.to_excel(writer, sheet_name='总体统计', index=False)
                logging.info("导出Sheet2: 总体统计")
                
                # Sheet3: 单位统计
                unit_stats.to_excel(writer, sheet_name='单位统计')
                logging.info("导出Sheet3: 单位统计")
            
            logging.info(f"结果导出成功: {output_file}")
            return True
        except Exception as e:
            logging.error(f"导出结果失败: {e}")
            return False
    
    def run(self, output_file: str = '匹配结果.xlsx') -> bool:
        """运行完整的匹配流程"""
        logging.info("=== 开始数据匹配流程 ===")
        
        try:
            # 1. 加载和清洗数据
            if not self.load_and_clean_data():
                return False
            
            # 2. 处理所有单位
            if not self.process_all_units():
                return False
            
            # 3. 导出结果
            if not self.export_results(output_file):
                return False
            
            logging.info("=== 数据匹配流程完成 ===")
            return True
        except Exception as e:
            logging.error(f"运行匹配流程时发生错误: {e}")
            return False

# 主程序入口
if __name__ == "__main__":
    # 创建数据匹配器实例
    matcher = DataMatcher(file_a='A.xlsx', file_b='B.xlsx')
    
    # 运行匹配流程
    success = matcher.run(output_file='数据匹配结果.xlsx')
    
    if success:
        logging.info("数据匹配任务已成功完成！")
    else:
        logging.error("数据匹配任务失败，请检查日志获取详细信息。")