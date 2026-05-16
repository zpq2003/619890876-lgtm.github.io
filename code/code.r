import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 图表全局样式设置 (符合顶级学术期刊出版标准)
# ==========================================
# 使用 seaborn 的 ticks 主题，干净简洁
sns.set_theme(style="ticks")

# 自定义 matplotlib 的底层参数
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans'] # 首选 Arial
plt.rcParams['axes.linewidth'] = 1.2      # 坐标轴边框加粗
plt.rcParams['xtick.direction'] = 'in'    # 刻度线向内
plt.rcParams['ytick.direction'] = 'in'    # 刻度线向内
plt.rcParams['xtick.top'] = True          # 显示上方刻度
plt.rcParams['ytick.right'] = True        # 显示右侧刻度
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# ==========================================
# 2. 模拟数据生成 (复现高度线性相关)
# ==========================================
np.random.seed(42) # 固定随机种子，确保每次运行结果一致，体现代码的可复现性
n_samples = 300

# 生成实际值 (Actual Values)
actual_values = np.random.uniform(10, 150, n_samples)

# 生成预测值 (Predicted Values)
# 构造高度线性相关的数据 (接近 R^2=0.95)，并加入符合特定偏差模式的噪声
noise = np.random.normal(0, 8, n_samples)
predicted_values = actual_values * 0.97 + noise - 1.5 

# ==========================================
# 3. 核心图表绘制
# ==========================================
fig, ax = plt.subplots(figsize=(7, 6), dpi=300) # 300 DPI 满足出版要求

# 绘制散点图
sns.scatterplot(
  x=actual_values, 
  y=predicted_values, 
  alpha=0.75,          # 透明度防止重叠遮挡
  edgecolor="black",   # 散点增加黑色边缘线，提升立体感
  linewidth=0.5,
  s=60,                # 散点大小
  color="#4C72B0",     # 经典的知性蓝
  ax=ax
)

# ==========================================
# 4. 绘制 1:1 对角基准线
# ==========================================
# 动态获取坐标轴的范围以适配不同数据
min_val = min(actual_values.min(), predicted_values.min()) - 10
max_val = max(actual_values.max(), predicted_values.max()) + 10

# 画出红色的 1:1 虚线
ax.plot(
  [min_val, max_val], 
  [min_val, max_val], 
  color='#C44E52',     # 采用稍暗的学术红
  linestyle='--', 
  linewidth=2, 
  zorder=0,            # 置于底层，避免遮挡散点
  label='1:1 Line'
)

# ==========================================
# 5. 坐标轴及标签优化
# ==========================================
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)

# 设置标签，加粗字体
ax.set_xlabel('Actual Values', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted Values', fontsize=14, fontweight='bold')

# 设置坐标轴刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=12)

# ==========================================
# 6. 右下角指标文本框设计
# ==========================================
# 准备文本内容 (使用 LaTeX 语法确保数学符号美观)
text_str = '\n'.join((
  r'$R^2 = 0.956$',
  r'$NSE = 0.955$',
  r'$KGE = 0.947$',
  r'$PBIAS = -1.445\%$'
))

# 文本框的样式属性
props = dict(
  boxstyle='round,pad=0.5', 
  facecolor='#F8F9FA', # 极浅的灰色背景
  alpha=0.9, 
  edgecolor='#B0B0B0', # 柔和的灰色边框
  linewidth=1
)

# 将文本框放置在图表右下角 (相对坐标体系：x=0.95, y=0.05)
ax.text(
  0.95, 0.05, 
  text_str, 
  transform=ax.transAxes, 
  fontsize=12,
  verticalalignment='bottom', 
  horizontalalignment='right', 
  bbox=props,
  fontfamily='monospace' # 等宽字体让数字对齐更整齐
)

# ==========================================
# 7. 收尾：调整布局与保存
# ==========================================
# 移除上方和右侧的边框线（可选：如果你更喜欢半包围结构的图）
# sns.despine(top=True, right=True) 

plt.tight_layout()

# 保存为高清 PDF 和 PNG (取消下方注释即可在本地保存)
# plt.savefig('model_prediction_scatter.pdf', format='pdf', bbox_inches='tight')
# plt.savefig('model_prediction_scatter.png', dpi=300, bbox_inches='tight')

# 展示图表
plt.show()