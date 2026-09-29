# 对比&目标
数据集：toolalpaca


teacher 模型本身的tool能力
原始qwen2.5


------ opd基线
glm: sft_on_better_A opd A 
SDFT: A opd self_A  -->  betterA   use ToolAlpaca数据
sft_on_better_A opd A plus 

------RL基线
RL + sft_on_better_A  / RL on betterA  这块还没想好
------ 原始基线
sft_on_better_A 
A


评测

通用能力

tool 能力

