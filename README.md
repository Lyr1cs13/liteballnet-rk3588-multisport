# LiteBallNet RK3588多球种追踪

面向网球、乒乓球和羽毛球的RK3588端侧时序热图追踪验证工程。

LiteBallNet以TrackNet的三帧时序热图追踪范式为原型：连续三帧画面拼接后输入网络，输出当前帧目标位置热图，再结合参考后处理得到稳定轨迹。本工程负责将算法侧交接的三套模型迁移至RK3588，并保持PC参考流程的轨迹表现和记录格式。

## 板端结果

在RK3588、RKNN Runtime2.3.2和RKNPU v2驱动0.9.8环境测试。以下为三路RKNN Runtime分别绑定NPU0、NPU1、NPU2时的**纯模型推理FPS**；不包含视频解码、缩放、热图解码、轨迹处理和视频写入。

|球种|测试窗口数|FP模型纯推理FPS|结果视频|
|---|---:|---:|---|
|网球|279|21.82|`outputs/tennis_reference_fp.mp4`|
|乒乓球|369|22.60|`outputs/table_tennis_reference_fp.mp4`|
|羽毛球|182|20.33|`outputs/badminton_reference_fp.mp4`|

结果视频右上角显示24窗口滑动纯推理FPS；每段视频配套逐帧坐标`CSV`和性能`JSON`。这些文件由板端FP模型重新推理生成，使用与交接PC脚本一致的候选点提取、时序稳定、补点、异常清理和轨迹绘制规则。

## 工程内容

```text
liteballnet_rk3588/
├── edge/       # RK3588运行器与参考后处理
├── models/     # 三球种PT、FP RKNN和INT8 RKNN模型
├── sources/    # 验证视频
├── outputs/    # 板端输出视频、CSV和JSON
├── tools/      # ONNX/RKNN转换与模型定义
└── docs/       # 算法侧交接的PC参考渲染脚本
```

## 板端运行

环境：Python3、OpenCV、NumPy、`rknn-toolkit-lite2`。

```bash
python3 edge/run_reference_pipeline.py --sport tennis --root .
python3 edge/run_reference_pipeline.py --sport table_tennis --root .
python3 edge/run_reference_pipeline.py --sport badminton --root .
```

运行器以当前帧、前一帧和前两帧组成`uint8[1,360,640,9]`的NHWC输入，调用对应FP RKNN模型，并输出MP4、CSV和JSON。乒乓球输入为120FPS，按交接演示策略以30FPS输出；其余视频按源帧率输出。

## PC转换与验证边界

转换在RKNN-Toolkit2.3.2环境完成：PT权重导出为固定尺寸ONNX，再生成FP和INT8 RKNN模型。正式效果视频使用FP模型，以优先保持算法侧参考输出；INT8模型仅用于性能探索。

`edge/run_reference_pipeline.py`是正式验证入口。`edge/run_liteballnet.py`与`edge/render_from_csv.py`保留为最小化INT8部署验证工具，不作为参考效果输出入口。

本仓库用于项目协作与板端验证，不替代球悟AI主工程中的训练分析和比赛判罚系统。模型权重、验证素材和转换产物仅限本私有仓库协作使用，不单独授权再分发。
