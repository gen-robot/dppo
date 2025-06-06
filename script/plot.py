import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
import json
import re

class ExperimentVisualizer:
    def __init__(self, log_dir, save_dir=None):
        """
        初始化可视化器
        
        Args:
            log_dir: 包含日志文件的目录
            save_dir: 保存图表的目录，默认与log_dir相同
        """
        self.log_dir = log_dir
        self.save_dir = save_dir if save_dir else log_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建数据存储字典
        self.data = {
            "success_rate": [],
            "avg_episode_reward": [],
            "avg_best_reward": [],
            "episode_whole_success_rate": [],
            "action_step_in_episode_success_rate": [],
            "steps": []
        }
        
        # 设置绘图样式
        plt.style.use('ggplot')
        
    def load_data(self, filename):
        """
        从日志文件加载数据
        
        Args:
            filename: 日志文件名
        """
        try:
            with open(os.path.join(self.log_dir, filename), 'r') as f:
                lines = f.readlines()
                
            # 定义正则表达式模式来匹配训练和评估数据
            # train_pattern = re.compile(r'train - (\d+): step\s+(\d+) \| .* \| episode_whole_success_rate:\s+([\d\.]+) \| avg episode reward - train: ([\d\.]+) \| avg best reward - train:\s+([\d\.]+)')
            # eval_pattern = re.compile(r'eval: success rate\s+([\d\.]+) \|avg episode reward ([\d\.]+) \| avg best reward\s+([\d\.]+) \| episode whole success rate\s+([\d\.]+) \| action step in episode success rate\s+([\d\.]+)')
            
            # 修改训练数据匹配模式
            train_pattern = re.compile(r'\[.*?\]\[.*?\]\[INFO\] - train - (\d+): step\s+(\d+) \|.*?\| episode_whole_success_rate:\s+([\d\.]+) \| avg episode reward - train:\s+([\d\.\-]+) \| avg best reward - train:\s+([\d\.\-]+)')
            # 修改评估数据匹配模式
            eval_pattern = re.compile(r'\[.*?\]\[.*?\]\[INFO\] - eval: success rate\s+([\d\.]+) \|avg episode reward\s+([\d\.\-]+) \| avg best reward\s+([\d\.]+) \| episode whole success rate\s+([\d\.]+) \| action step in episode success rate\s+([\d\.]+)')
            
            for line in lines:
                # 尝试匹配训练数据行
                train_match = train_pattern.search(line)
                if train_match:
                    iteration = int(train_match.group(1))
                    step = int(train_match.group(2))
                    episode_whole_success_rate = float(train_match.group(3))
                    avg_episode_reward = float(train_match.group(4))
                    avg_best_reward = float(train_match.group(5))
                    
                    self.data['steps'].append(step)
                    self.data['episode_whole_success_rate'].append((step, episode_whole_success_rate))
                    self.data['avg_episode_reward'].append((step, avg_episode_reward))
                    self.data['avg_best_reward'].append((step, avg_best_reward))
                    continue
                
                # 尝试匹配评估数据行
                eval_match = eval_pattern.search(line)
                if eval_match:
                    success_rate = float(eval_match.group(1))
                    avg_episode_reward = float(eval_match.group(2))
                    avg_best_reward = float(eval_match.group(3))
                    episode_whole_success_rate = float(eval_match.group(4))
                    action_step_in_episode_success_rate = float(eval_match.group(5))
                    
                    # 对于评估数据，我们使用当前步骤作为step值（假设这是在训练步骤之后的评估）
                    step = 0  # 默认值
                    if len(self.data['steps']) > 0:
                        step = self.data['steps'][-1]  # 使用最近的训练步骤
                    
                    self.data['success_rate'].append((step, success_rate))
                    self.data['action_step_in_episode_success_rate'].append((step, action_step_in_episode_success_rate))
                    
                    # 如果使用第一个评估步骤（还没有训练步骤），则添加到steps列表
                    if step not in self.data['steps'] and len(self.data['steps']) == 0:
                        self.data['steps'].append(step)
                    
            # 去重并排序steps
            self.data['steps'] = sorted(list(set(self.data['steps'])))
            
            # 排序所有指标数据
            for key in self.data:
                if key != 'steps':
                    self.data[key].sort(key=lambda x: x[0])
            
            print(f"成功加载数据，共{len(self.data['steps'])}个步骤")
            
            # 打印一些数据统计信息
            for key in self.data:
                if key != 'steps':
                    print(f"{key}: {len(self.data[key])}个数据点")
                    
        except Exception as e:
            print(f"加载数据时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def convert_to_dataframe(self):
        """将数据转换为DataFrame以便于处理"""
        df_dict = {'step': self.data['steps']}
        
        for key in self.data:
            if key != 'steps' and self.data[key]:
                # 创建一个与steps长度相同的列表，填充NaN
                values = [np.nan] * len(self.data['steps'])
                
                # 将数据填入正确的位置
                for step, value in self.data[key]:
                    if step in self.data['steps']:
                        idx = self.data['steps'].index(step)
                        values[idx] = value
                
                df_dict[key] = values
        
        return pd.DataFrame(df_dict)
    
    def interpolate_missing_values(self, df):
        """插值处理缺失值以获得连续的线条"""
        # 使用前向填充和后向填充，确保连续性
        interpolated_df = df.copy()
        for column in df.columns:
            if column != 'step':
                # 只对有数据的列进行插值
                if not df[column].isnull().all():
                    # 线性插值
                    interpolated_df[column] = df[column].interpolate(method='linear')
        
        return interpolated_df
    
    def plot_metrics(self):
        """绘制所有指标的图表"""
        df = self.convert_to_dataframe()
        
        # 绘制所有指标
        metrics = [
            'success_rate', 
            'avg_episode_reward', 
            'avg_best_reward', 
            'episode_whole_success_rate', 
            'action_step_in_episode_success_rate'
        ]
        
        for metric in metrics:
            if metric in df.columns and not df[metric].isnull().all():
                self.plot_single_metric(df, metric)
        
        # 绘制成功率相关指标的对比图
        success_metrics = [m for m in [
            'success_rate', 
            'episode_whole_success_rate', 
            'action_step_in_episode_success_rate'
        ] if m in df.columns and not df[m].isnull().all()]
        
        if len(success_metrics) > 1:
            self.plot_comparison(df, success_metrics, 'success_rates_comparison')
        
        # 绘制奖励相关指标的对比图
        reward_metrics = [m for m in [
            'avg_episode_reward', 
            'avg_best_reward'
        ] if m in df.columns and not df[m].isnull().all()]
        
        if len(reward_metrics) > 1:
            self.plot_comparison(df, reward_metrics, 'rewards_comparison')
    
    def plot_single_metric(self, df, metric):
        """
        绘制单个指标的图表
        
        Args:
            df: 包含数据的DataFrame
            metric: 要绘制的指标名称
        """
        if df[metric].isnull().all():
            print(f"跳过绘制 {metric}，没有有效数据")
            return
        
        # 获取有效数据点的索引
        valid_indices = df[metric].notna()
        valid_df = df[valid_indices]
            
        plt.figure(figsize=(10, 6))
        
        # 绘制连线+标记点
        plt.plot(valid_df['step'], valid_df[metric], 
                marker='o', linestyle='-', linewidth=2, markersize=6,
                markerfacecolor='white', markeredgewidth=2)
        
        plt.title(f'{metric.replace("_", " ").title()} vs Training Steps')
        plt.xlabel('Training Steps')
        plt.ylabel(metric.replace("_", " ").title())
        plt.grid(True, alpha=0.3)
        
        # 设置x轴为整数
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 自动添加数据点标签，但避免过度拥挤
        step = max(1, len(valid_df) // 8)  # 降低标签密度
        for i in range(0, len(valid_df), step):
            plt.annotate(f'{valid_df[metric].iloc[i]:.3f}', 
                        (valid_df['step'].iloc[i], valid_df[metric].iloc[i]),
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{metric}.png'), dpi=300)
        plt.close()
        print(f"已保存 {metric} 图表")
    
    def plot_comparison(self, df, metrics, filename):
        """
        绘制多个指标的对比图
        
        Args:
            df: 包含数据的DataFrame
            metrics: 要对比的指标列表
            filename: 保存的文件名
        """
        plt.figure(figsize=(12, 7))
        
        for metric in metrics:
            if not df[metric].isnull().all():
                # 获取有效数据点
                valid_indices = df[metric].notna()
                valid_df = df[valid_indices]
                
                plt.plot(valid_df['step'], valid_df[metric], 
                        marker='o', linestyle='-', linewidth=2, markersize=6,
                        label=metric.replace("_", " ").title())
        
        plt.title('Metrics Comparison')
        plt.xlabel('Training Steps')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.8)
        
        # 设置x轴为整数
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{filename}.png'), dpi=300)
        plt.close()
        print(f"已保存 {filename} 对比图表")
    
    def generate_statistics(self):
        """生成数据统计表"""
        df = self.convert_to_dataframe()
        
        # 计算统计数据
        stats = {}
        metrics = [
            'success_rate', 
            'avg_episode_reward', 
            'avg_best_reward', 
            'episode_whole_success_rate', 
            'action_step_in_episode_success_rate'
        ]
        
        for metric in metrics:
            if metric in df.columns and not df[metric].isnull().all():
                stats[metric] = {
                    'mean': df[metric].mean(),
                    'max': df[metric].max(),
                    'min': df[metric].min(),
                    'std': df[metric].std(),
                    'last_value': df[metric].iloc[-1] if not np.isnan(df[metric].iloc[-1]) else 'N/A'
                }
        
        # 创建统计表
        stats_df = pd.DataFrame(stats).T
        stats_df.to_csv(os.path.join(self.save_dir, 'statistics.csv'))
        print("已保存统计数据")
        
        return stats_df
    
    def run_visualization(self, log_filename):
        """运行完整的可视化流程"""
        print(f"开始处理日志文件: {log_filename}")
        self.load_data(log_filename)
        self.plot_metrics()
        stats = self.generate_statistics()
        print("\n数据统计:")
        print(stats)
        print("\n可视化完成！图表已保存到:", self.save_dir)


if __name__ == "__main__":
    # 使用示例
    log_dir = "/home/zhouzhiting/Projects/dppo/sapien-finetune/drawer_ppo_diffusion_ta20_td50_tdf5/2025-06-05_17-48-16_42"  # 修改为您的日志目录
    log_filename = "run.log"  # 修改为您的日志文件名
    
    visualizer = ExperimentVisualizer(log_dir)
    visualizer.run_visualization(log_filename)