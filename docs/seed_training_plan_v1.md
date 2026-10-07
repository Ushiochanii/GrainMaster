# 小麦种子实例分割：第一轮训练准备与计划

日期：2026-10-07。下文保留数据准备时的计划；最新执行状态见文末。

## 实际数据

来源：`data/seed.coco-segmentation.zip`。172 张单培养皿图片，来自 36 张原图，6,983 个实例。
导出有两个同名 seed 类别条目，但所有实例使用类别 1；已统一为 YOLO 类别 0：seed。
COCO 使用压缩 RLE mask，并非检测框。逐实例解码后转为外轮廓多边形，没有额外简化。
所有实例转换 IoU ≥ 0.98343887；少量内部孔洞或孤立像素不能由单条外轮廓完整表达。
这个 IoU 是格式转换一致性，不是模型精度，也不是标注正确性的证明。
原始 COCO 包完整保留，最终测量误差评价应以其原始 mask 为参考。
172 张导出图片的解码像素与原单皿导出完全一致。
标签来自用户提交并声明已完成的人工/SAM 辅助标注；实际边界正确性仍需人工抽查。

## 已固定的分组

按原始照片分组，随机种子 42；26 张原图训练、5 张验证、5 张测试。
同一原图的所有培养皿固定在同一组。Roboflow 中全部 Train 的分配未被沿用。

| 用途 | 原图组数 | 培养皿图 | 种子实例 |
|---|---:|---:|---:|
| 训练 | 26 | 116 | 4,422 |
| 验证 | 5 | 19 | 827 |
| 测试 | 5 | 22 | 986 |
| 待复核，不参与训练/评分 | 涉及 10 张原图 | 15 | 748 |

复核区的原图组数与训练/验证/测试有重叠，仅代表被隔离的皿来自这些照片；没有进入任何模型数据加载路径。
补标后必须回到 metadata 中的 planned_split，不能重新随机分组。
验证原图：0501、0513、0508、0504、0509。
测试原图：0466、0473、0461、0510、0516。
0462 属于训练集，后续其演示结果不能用作独立模型精度证据。
若不同文件实际是同一批物理种子的重复拍摄，还需要按物理样品进一步合并分组；目前没有这类样品对应信息。

## 复核与补标

52 个实例 mask 碰到图像边界，共涉及 15 张皿图。边界接触是风险提示，不自动等于标错。
为避免截断种子影响面积、周长，目前保守隔离整张皿图，而不是只删掉个别标签后把种子当背景。
此前指出的五张裁剪失败均包含其中。其他十张也需检查是否为裁剪截断或边缘标注误差。
修复前不能把标签直接贴到更大的新裁剪图上，必须按原图偏移迁移，再补上新增区域的种子。
文件列表与具体实例 ID 见 boundary_review.csv；全部旧图和标签在 review 下保留。

15 张：0467_dish01、0467_dish05、0484_dish03、0484_dish04、0501_dish01、
0501_dish02、0503_dish05、0504_dish04、0509_dish01、0509_dish05、0510_dish01、
0516_dish05、0518_dish04、0518_dish05、0519_dish02。

## 训练方案

固定 YOLO26s-seg，预训练权重初始化，imgsz=1024，单类别 seed，原始颜色/几何的单皿输入。
第一轮只训练此模型，不开展尺寸或分辨率 benchmark。

提议最多 150 epochs，patience=30，AdamW，初始学习率 0.001，cosine 衰减，warmup 3 epochs。
batch=4 起步，显存不足时减到 2；mask_ratio=1 保留更细的训练标签，overlap_mask=false。
mask_ratio 不会改变模型原生预测头分辨率，也不能保证细节准确。
旋转、翻转和小幅尺度/位置变化；仅轻微饱和度/亮度变化，不改变色相。
关闭 mosaic、mixup、copy-paste、透视和剪切变换，先建立容易解释的基线。
参数在 configs/train_seed_v1.yaml，可通过 --plan-config 使用；需先安排训练设备。

顺序：
1. 抽查标注总览，确认粘连种子逐粒分开、轮廓不包含阴影/蓝色背景、没有明显漏标。
2. 处理复核区，或明确保持隔离后开始首轮；不可用不完整标签覆盖新裁剪图。
3. 确定计算设备后运行少量训练步骤检查损失、标签显示和显存；此步骤目前未执行。
4. 正式训练，仅用验证集选模型和阈值；保留 best.pt、last.pt、训练曲线与配置。
5. 配置冻结后进行独立测试，避免反复依据测试集调参。

## 计算设备

本地显卡：AMD Radeon RX 7800 XT。当前 torch=2.14.1+cpu，CUDA 不可用。
Ultralytics=8.4.174。尚未安装/验证 AMD 训练后端。
CPU 可以做数据检查与小规模试运行，但不建议作为完整 1024 训练的首选。
优先安排可用的受支持 GPU 训练环境；也可评估本地 AMD 方案，但需要另行验证环境兼容性。
本次没有租用云 GPU、改变 PyTorch 环境、下载新模型或启动训练。

## 评价与接受标准

识别：mask mAP50、mAP75、mAP50–95，逐实例匹配后的 precision/recall，逐皿计数误差及漏检、合并、拆分例图。
轮廓：IoU/Dice，以及边界距离中位数、P90（原始像素）；区分独立、接触、遮挡实例。
测量：原始 COCO GT mask 与 prediction 使用同一空间变换及同一测量函数，比较长、宽、面积和周长的绝对误差、相对误差及偏差。
只在一一匹配且完整可见的种子上报告完整形态误差，同时明确匹配覆盖率和漏检，不能只报告成功实例。
长度/宽度/面积沿用核心 phenotyping；周长评价使用同一固定轮廓/平滑规则，对 GT 与预测一致处理。
周长评价工具尚需在训练结果出来前实现与核查，不能将其描述为已完成验证。
物理尺度由长尺负责，比较 mask 引起的相对几何误差不等于验证真实毫米尺度。
可将长、宽、面积的相对误差中位数 ≤5% 作为首轮讨论目标，并报告 P90 与有符号偏差；不是已达到的结果。
周长单独验证，不从面积精度推断周长精度。必要时先不把周长作为正式生物学表型。
测试仅 5 张原图，应按原图分组汇总/估计不确定性，不能把每粒种子当独立拍摄重复。

## 文件与复现

数据：data/processed/seed_yolo_v1/{images,labels}/{train,val,test}。
隔离区：data/processed/seed_yolo_v1/review/{images,labels}。
dataset.yaml、metadata.json、manifest.csv、boundary_review.csv 和 qc 总览均已生成。
转换脚本：scripts/prepare_roboflow_coco.py；依赖 pycocotools（annotation 可选依赖）。

仅检查，不训练：
```powershell
python scripts/train_yolo26.py --data data/processed/seed_yolo_v1/dataset.yaml --check-only
```

后续经确定设备和复核安排后使用的训练命令（本次未执行）：
```powershell
python scripts/train_yolo26.py --data data/processed/seed_yolo_v1/dataset.yaml --plan-config configs/train_seed_v1.yaml --device <确定的设备>
```

依据：[Ultralytics 分割数据格式](https://docs.ultralytics.com/datasets/segment/)；
[官方训练参数说明](https://docs.ultralytics.com/usage/cfg/)。

## 执行更新：用户已授权 WSL/ROCm 训练

已找到 /root/.venvs/wheat_awn_rocm 的 ROCm PyTorch 环境。
通过 HSA_ENABLE_DXG_DETECTION=1 启用 WSL GPU 检测后，RX 7800 XT 前向/反向及有限梯度检查通过。
PyTorch 2.9.1+rocm7.2.1，约 16 GB 显存；隔离环境 .venv-wsl 中安装 Ultralytics 8.4.174，
只读复用原环境的 ROCm 包，不替换麦芒项目的环境。
Linux 数据路径写入 dataset_wsl.yaml，Windows 的 dataset.yaml 保留。
用户已授权开始训练；后台执行器先做 1 epoch / 15% 训练图的 smoke，验证集保持全部 19 张，
成功退出后自动开始 150 epoch 上限、patience=30 的正式训练。
执行结果：smoke 完成训练、验证及 best.pt/last.pt 保存；正式训练已于 05:22 UTC 启动，
已观察到第 1/150 epoch 的 GPU 批次和有限损失值。
首次采用 FP32（amp=false）减少 ROCm 精度兼容变量。
首次 smoke 在 batch=4 和 batch=2 触发显存不足，框架自动降至 1；正式配置已同步改为 batch=1，保留 1024 分辨率和 mask_ratio=1。
测试集未参与 smoke 或训练。
当前阶段和错误由 artifacts/models/seed_v1/run_status.json 记录；日志在 training.log。
正式模型目录 artifacts/models/training/seed_v1_rocm，权重在 weights/best.pt 和 last.pt。
结果尚未生成前不能宣称模型达到任何目标精度；以状态文件和训练日志为准。
## 执行更新：AMP 与数据加载（2026-10-07）

用户已确认此 GPU 环境能使用 AMP，并授权停止旧训练、尝试 batch=2 和更多 worker。
旧 FP32 训练在第 1 轮第 17/116 批停滞，已停止对应 WSL 进程组；旧日志与模型目录归档到
artifacts/models/seed_v1/archive_fp32_20261007，未删除原记录。
新配置 amp=true、batch=2、workers=2，保持 1024、mask_ratio=1、overlap_mask=false。
移除执行器强制关闭 AMP 的覆盖；先用全部 116 张训练图完成 1 epoch 检查（验证 19 张），
覆盖高密度培养皿，再开始正式训练。检查的成功、实际 batch/AMP 和速度以新日志为准。
若 Ultralytics 自动降低 batch，不得将请求的 batch=2 当作实际有效 batch=2。

后续检查：Ultralytics 自带的 FP16 COCO 检测一致性检查失败，自动关闭 AMP；该轮已停止并归档。
检查失败不能据此断言所有混合精度均不可用。改用官方支持的 amp=bf16；RX7800XT 的
BF16 支持检测、矩阵运算及有限反向梯度检查通过。channels_last=false 避免自动套用 CUDA 内存布局。
训练入口新增 runtime_settings.json，记录实际 amp_enabled、amp_mode、batch、workers；
拒绝请求 AMP 后静默回退 FP32，每批检查有限损失。首轮完成后同步实际 batch 到正式配置。
当前首轮运行实际确认 amp_enabled=true、amp_mode=bf16、batch=2、workers=2；
完整首轮成功与最终精度仍须以真实结果为准。

实际试验：BF16 + batch=2 在前几批出现 OOM，Ultralytics 自动回退 batch=1；
正式配置同步为 batch=1。首轮 runtime_settings.json 的 batch 已根据明确的 OOM 回退日志
修正为 1（batch_source 记录来源）；后续入口在训练结束再次记录最终生效 batch。
有限损失回调已适配 Ultralytics 8.4.174 的字典格式，初次格式错误导致的中断已归档。
当前不能声称 batch=2 可稳定训练，也不能声称 BF16 已通过整个数据集的长期稳定性验证。

BF16 首轮：116 张训练图全部完成，训练日志计时 2:32（含 batch=2 OOM 与回退重试）。
验证仍按默认 train batch 的两倍加载；在密集种子损失阶段 OOM，本轮未完成验证或保存权重。
训练入口 SeedTrainer 限制验证 batch=1，再按 train batch=1 重新检查完整首轮；
不降低 imgsz 或 mask_ratio。此时仍不能声称检查完成。

最终检查通过：BF16、train batch=1、val batch=1、workers=2 完成全部 116 张训练图和 19 张验证图，
保存 smoke best.pt/last.pt，并完成最终权重验证。results.csv 第 1 轮 time=181.798 秒；
训练批次日志计时 1:49。首轮 mask mAP50=0.36721、mAP50-95=0.09829 仅说明已有有效评估，
不代表最终模型精度。正式 150 epoch / patience=30 已启动；实际运行设置由 runtime_settings.json
确认 amp_enabled=true、amp_mode=bf16、batch=1、workers=2。测试集仍未使用。

## 恢复训练与 Windows 原生识别检查

第 3 轮 60/116 批日志长期停滞，两次调用栈采样均位于 segmentation loss；
没有明确 HIP/CUDA OOM 报错，不能将此次停滞等同已证实的显存溢出。
用户授权后停止旧进程组，保留 stalled_epoch3.log，并从 last.pt 恢复第 3 轮，
保留模型及优化器进度（第 3 轮未保存的部分重新计算）。新增 --resume 入口；
恢复模式跳过 smoke，沿用正式模型目录、日志与监视面板。
独立 Windows 环境位于 .venv-rocm-win；安装包缓存位于 artifacts/windows_rocm_probe/pip_cache。
原有 Windows CPU PyTorch 与 WSL 环境均不替换。Windows 首次检查仅枚举 GPU，
不创建模型或 tensor，不能据此宣称训练/混合精度已经验证可用。

Windows 原生识别检查实测成功：独立环境 Python 3.13.13、torch 2.13.0+rocm10.0.0、
HIP 7.15.26333，在 Windows 11 build 22631 上返回 gpu_available=true，设备 RX7800XT。
仅设备枚举，无 tensor/模型创建；报告 artifacts/windows_rocm_probe/device_report.json。
这验证了原生 GPU 可识别，不等于验证 YOLO 训练、BF16 或长期稳定性。
安装来源 AMD TheRock stable index，gfx1101 对应包；没有更新显卡驱动或系统。
WSL 恢复后成功完成第 3 至第 6 轮，进入第 7 轮后再次日志停滞，无显式 OOM 报错。
用户要求继续训练，因此保留 stalled_epoch7.log，再从已保存的第 6 轮 last.pt 恢复第 7 轮。
间歇性停滞的根因尚未解决，不能将重启后推进当作彻底修复。

## Windows 真实训练测试未通过：系统蓝屏重启

2026-10-07 用户报告电脑意外关闭。事件 1001 确认 bugcheck 0x9F，参数1=3，
对应 DRIVER_POWER_STATE_FAILURE / 电源请求长时间未完成。具体故障驱动未定位。
系统转储 C:\Windows\Minidump\100726-8625-01.dmp 已存在；未上传或修改转储。
真实训练测试仅推进到第17/116批，GPU_mem 日志最高22.4G；不能将此数值等同物理显存用量，
也不能单凭此判定 OOM。Windows smoke 未完成，不得转入正式训练或宣称原生更稳定。
重启后检查没有残留 train_yolo26 / launch_seed_training 进程；已停止所有 GPU 测试，
未自动重启训练。原 WSL 第6轮检查点与 results.csv 保留。
系统事件摘录 artifacts/windows_rocm_probe/system_crash_events.json；run_status 标记 failed。

用户随后明确授权继续运行。恢复选择 WSL/ROCm 原环境，从原 seed_v1_rocm/weights/last.pt
保存的第6轮恢复第7轮，BF16 / batch=1 / workers=2 / 1024 / mask_ratio=1 均保持。
未重新启动曾蓝屏的 Windows 真实训练测试，未更改驱动或电源设置。
当前间歇性停滞原因仍未确定；重新运行不代表根因已修复。监视面板已恢复 WSL 状态。

## 用户授权：OOM 与日志停滞自动恢复

WSL 正式训练启用进程守护：仅明确的 fatal HIP/CUDA OOM 或连续120秒无新增日志触发恢复（每次启动前180秒为宽限期）；
最多3次自动重试（加首次运行共4次），次数为本次执行器启动的总上限，不随轮次重置。
先终止训练进程及同组数据加载 worker，等待10秒，再从最近保存的 last.pt 恢复模型和优化器。
没有检查点则停止，普通代码/数据错误及非有限损失不重试；达到上限标记 failed。
手动终止执行器时会清理其独立训练进程组，不会将手动终止当作自动恢复事件。
Windows 蓝屏后没有启用 Windows 自动恢复，也不会在系统重启后自动启动任何训练。
状态文件记录 retry_count、retry_limit、trainer_pid、stall_timeout_seconds；
recovery_history.jsonl 记录恢复原因、次数和时间；监视面板显示等待恢复及次数。
验证使用小型无GPU进程：OOM恢复、超时重试上限、普通错误、数据worker清理共5项通过；
监视接口4项通过。守护已接管真实WSL训练，从已保存第13轮恢复第14轮，未人为制造真实OOM。


## 保持 mask 精度的省显存模式（2026-10-07）

用户选择分块 mask 损失与梯度检查点方案。`--memory-safe` 将匹配实例的
mask 损失每8个一组计算，在反向传播时重新计算组内中间结果。
这是损失计算部分的检查点，不是整个骨干网络的检查点。
保持 YOLO26s-seg、imgsz=1024、mask_ratio=1、batch=1、workers=2、BF16，
不改变损失定义、标注或优化器；只在当前进程替换计算函数，不修改安装的 Ultralytics。
代价是增加反向传播计算量，不能保证驱动层停滞不会再出现。

普通模式连续3次恢复后仍失败，会额外尝试一次省显存模式；该备用尝试再失败就停止。
若启动时已指定省显存模式，则沿用最多3次恢复，不再叠加相同备用尝试。
OOM、停滞、恢复检查点、进程组清理规则与上述守护一致。

验证：损失及梯度一致性、重叠/独立 mask、无梯度验证路径、BF16有限性已检查；
训练守护6项、损失3项、监视接口4项，共13项单元测试通过。
GPU合成测试（128个256×256的mask）中，损失部分峰值分配内存由172045824字节
降至83912192字节。报告：artifacts/models/seed_v1/memory_loss_check.json。
此测试仅涉及mask损失，不代表整模型显存降幅或训练速度。
真实WSL训练已从第13轮检查点恢复并完成第14轮116个训练批次；
日志显示GPU_mem最高约8.39G，仍需持续观察验证和后续轮次的稳定性。
