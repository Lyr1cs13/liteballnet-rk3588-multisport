# LiteBallNet RK3588 Multi-Sport Tracking

RK3588端侧多球种轻量时序热图追踪验证工程。工程接收三套针对网球、乒乓球、羽毛球的LiteBallNet权重与对应视频片段，在PC转换环境完成ONNX导出和INT8 RKNN量化，在RK3588端完成三核并行推理、轨迹可视化和结果视频输出。

## 验证结果

测试使用RK3588、RKNN Runtime2.3.2、RKNPU v2驱动0.9.8；输入为三帧BGR时序拼接，模型输入语义为9通道`360x640`。FPS为三个独立RKNN Runtime分别绑定NPU0/1/2后的纯模型推理吞吐，不包含视频解码、缩放、热图解码、轨迹绘制和写视频。

|模型|测试片段|纯推理FPS|输出视频|
|---|---:|---:|---|
|网球LiteBallNet-INT8|279个三帧窗口|35.98|`outputs/tennis_rknn.mp4`|
|乒乓球LiteBallNet-INT8|369个三帧窗口|36.10|`outputs/table_tennis_rknn.mp4`|
|羽毛球LiteBallNet-INT8|182个三帧窗口|36.28|`outputs/badminton_rknn.mp4`|

输出画面右上角显示24窗口滑动纯推理FPS，轨迹使用热图候选峰值、时序关联和无检测断开策略生成。每段视频均配套`CSV`逐帧坐标记录及`JSON`性能结果。

## 工程结构

```text
liteballnet_rk3588/
├── edge/                 # RK3588运行器与无推理重绘工具
├── models/               # 三类INT8 RKNN模型与原始PT权重
├── sources/              # 三个验证片段
├── outputs/              # 板端生成的可视化视频、坐标CSV、性能JSON
├── tools/                # 模型定义与PC侧ONNX/RKNN转换脚本
└── docs/                 # 交接渲染参考实现与输出关键帧
```

## 端侧运行

板端环境需要Python3、OpenCV、NumPy和`rknn-toolkit-lite2`。将工程同步到RK3588后，在工程根目录执行：

```bash
python3 edge/run_liteballnet.py --sport tennis --root .
python3 edge/run_liteballnet.py --sport table_tennis --root .
python3 edge/run_liteballnet.py --sport badminton --root .
```

`run_liteballnet.py`会：

1. 以当前帧、前一帧、前两帧构成9通道BGR时序输入；
2. 建立三套RKNN Runtime并分别绑定NPU三核；
3. 运行INT8热图模型并以完成时间计算滑动纯推理FPS；
4. 从热图峰值提取球心候选，按时间连续性生成轨迹；
5. 输出MP4、CSV和JSON结果。

为不重复运行NPU、仅调整展示样式，可基于已生成CSV重绘：

```bash
python3 edge/render_from_csv.py \
  --video sources/tennis.mp4 \
  --csv outputs/tennis_rknn.csv \
  --output outputs/tennis_rknn.mp4
```

乒乓球输入片段为120FPS，演示输出固定为30FPS；其余按源视频帧率输出。

## PC侧转换

`tools/export_models.py`包含与交接模型一致的轻量DS-U-Net与坐标注意力结构定义。转换环境为RKNN-Toolkit22.3.2：加载严格匹配的PT状态字典，导出固定shape ONNX，按对应运动片段抽取校准样本，生成FP和INT8 RKNN模型。

```bash
python tools/export_models.py --bundle-root /path/to/LiteBallNet_demo_bundle
```

转换脚本的输入归一化配置为`mean=0,std=255`，保持PC端原始实现的`/255`语义；板端优化图按NHWC传入`uint8[1,360,640,9]`，避免Runtime进行隐式布局转换。

## 说明

- `docs/pc_reference_renderer.py`为算法同学交接的PC参考渲染实现，保留作结果对照；端侧运行入口为`edge/run_liteballnet.py`。
- 模型权重、验证片段和转换产物仅限本私有仓库的项目协作与验证使用，不单独授权再分发。
- 该工程是多球种时序热图模型的板端迁移验证，不替代球悟AI主工程中的训练分析和比赛判罚系统。
