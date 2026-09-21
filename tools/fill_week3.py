from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

SOURCE = Path(r"C:\\Users\\97689\\AppData\\Local\\Temp\\week3_blank.docx")
OUTPUT = Path(r"C:\\Users\\97689\\AppData\\Local\\Temp\\week3_liteballnet.docx")
MONTAGE = Path(r"C:\\Users\\97689\\AppData\\Local\\Temp\\liteballnet_results.png")


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill)
    tc_pr.append(shd)


def set_run(run, bold=False, size=10.5):
    run.font.name = '宋体'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    run.font.size = Pt(size)
    run.bold = bold


def add_text(cell, text, bold=False, after=6, size=10.5):
    paragraph = cell.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.16
    run = paragraph.add_run(text)
    set_run(run, bold, size)
    return paragraph


def main():
    document = Document(SOURCE)
    document.paragraphs[0].text = '深圳技术大学项目实践周记'
    document.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in document.paragraphs[0].runs:
        run.font.name = '黑体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        run.font.size = Pt(16)
        run.bold = True
    document.paragraphs[1].text = '第3周周记：2026年9月21日~2026年9月27日'
    document.paragraphs[1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in document.paragraphs[1].runs:
        set_run(run, size=10.5)

    table = document.tables[0]
    table.cell(2, 1).text = '多球种轻量时序热图追踪模型的RK3588端侧迁移验证'
    for cell in table.rows[2].cells:
        for para in cell.paragraphs:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.font.name = '宋体'
                run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
                run.font.size = Pt(10.5)
                run.bold = True

    cell = table.cell(3, 0)
    cell.text = ''
    cell.vertical_alignment = 1
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(7)
    r = p.add_run('一、工作内容')
    set_run(r, bold=True, size=11)

    add_text(cell, '本周接收算法方向同学交接的网球、乒乓球、羽毛球三套LiteBallNet权重、对应视频片段及PC参考后处理，完成面向RK3588的统一迁移验证。该项工作用于验证轻量时序热图追踪模型在不同小球场景下的板端可运行性，不替代原网球训练分析主工程。')
    add_text(cell, '1.梳理模型输入输出与转换约束。三套模型均以当前帧、前一帧、前两帧的BGR图像拼接为9通道时序输入，固定尺寸为360×640，输出单通道球体响应热图。依据交接代码复现轻量深度可分离U-Net及坐标注意力结构，并使用严格状态字典加载三个运动的独立权重。', bold=False)
    add_text(cell, '2.完成PC侧模型转换。使用RKNN-Toolkit2.3.2，将三套PT权重分别导出ONNX，并从各自视频中按间隔采样生成校准数据，完成FP与INT8 RKNN转换。正式参考结果选用FP模型，优先保持PC端热图数值与候选结果的一致性；针对源模型NCHW语义与RKNN优化图NHWC输入的差异，板端统一传入uint8[1,360,640,9]，消除了Runtime逐帧隐式布局转换。')
    add_text(cell, '3.完成RK3588端侧并行推理与参考后处理还原。每套模型创建3个独立RKNN Runtime，分别绑定NPU Core0/Core1/Core2；以三帧窗口加载视频，按完成时间计算24窗口滑动纯推理FPS。除RKNN推理后端和性能标签外，直接复用交接PC脚本的候选峰值提取、时序稳定、尖峰清理、受限插值、短片段剔除与红色渐隐绘制。')
    add_text(cell, '4.完成三段参考效果可视化输出。输出视频右上角仅显示滑动纯推理FPS，其他轨迹表现与交接PC参考实现保持一致；同步保存逐帧坐标CSV和性能JSON，便于后续与PC结果复核。')

    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(MONTAGE), width=Cm(14.7))
    caption = cell.add_paragraph()
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_after = Pt(7)
    r = caption.add_run('图1 三类运动片段的RK3588端侧FP参考还原结果，右上角为滑动纯推理FPS')
    set_run(r, size=9)

    add_text(cell, '二、测试结果', bold=True, size=11)
    result = cell.add_paragraph()
    result.paragraph_format.space_after = Pt(7)
    result.paragraph_format.line_spacing = 1.16
    for text, bold in [
        ('网球：279个三帧窗口，FP纯推理21.82FPS；', True),
        ('乒乓球：369个三帧窗口，FP纯推理22.60FPS；', True),
        ('羽毛球：182个三帧窗口，FP纯推理20.33FPS。', True),
        ('上述统计仅覆盖三路RKNN Runtime模型推理，排除视频解码、缩放、热图解码、轨迹绘制及视频写入。', False),
    ]:
        run = result.add_run(text)
        set_run(run, bold=bold)

    add_text(cell, '三、心得体会', bold=True, size=11)
    add_text(cell, '本周的关键不在于单纯完成格式转换，而在于保持时序输入、归一化、热图解码和轨迹稳定化规则的一致性，并区分模型吞吐与完整视频处理帧率。三类片段的结果表明，统一的转换与运行框架能够承载不同小球项目的独立权重；后续应在更长视频和真实采集画面上继续验证复杂遮挡、快速移动与场景切换下的稳定性。')

    sign = cell.add_paragraph()
    sign.paragraph_format.space_before = Pt(8)
    sign.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = sign.add_run('填写人：李睿鑫    2026年9月27日')
    set_run(run, size=10.5)
    document.save(OUTPUT)


if __name__ == '__main__':
    main()
