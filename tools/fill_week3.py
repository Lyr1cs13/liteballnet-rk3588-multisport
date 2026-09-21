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
    table.cell(2, 1).text = '基于TrackNet思路的多球种轻量追踪模型板端部署测试'
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

    add_text(cell, '本周主要配合算法同学钊沣完成LiteBallNet的板端验证。钊沣交接了网球、乒乓球、羽毛球三套权重、测试视频和PC参考脚本；该模型以TrackNet的连续三帧输入与热图定位思路为原型，采用更轻量的网络结构。本周我的工作是将PC侧流程迁移到RK3588，保证模型输入、追踪规则和显示效果能够对应。')
    add_text(cell, '1.先核对PC侧流程。模型每次取当前帧、前一帧和前两帧，缩放后拼成9通道输入，输出一张球位置热图。钊沣的脚本会从热图中找候选点，再处理短暂丢球、错误跳点和轨迹断开，最后画出红色渐隐拖尾。我先按这个流程核对了板端需要保留的部分，避免只跑通模型却让展示效果变样。')
    add_text(cell, '2.完成模型转换与输入适配。使用PC虚拟机中的RKNN-Toolkit2.3.2，将三套PT权重导出ONNX并转换为RKNN。转换后发现模型在板端运行时需要按NHWC方式传入9通道数据；调整输入布局后，消除了Runtime逐帧自动转置带来的额外开销。正式效果测试采用FP模型，优先保证与PC热图结果接近；INT8版本仅保留作性能探索。')
    add_text(cell, '3.完成板端运行器。为三套模型各建立3个RKNN Runtime，分别绑定RK3588的NPU三核并行处理三帧窗口。板端只替换了原脚本中的模型推理部分，其余候选提取、轨迹稳定、补点、异常点清理和红色拖尾绘制均沿用钊沣的参考实现；右上角额外显示24窗口滑动纯推理FPS。')
    add_text(cell, '4.完成三段视频输出。每段视频同时保存轨迹坐标CSV和测试JSON，后续可直接和PC结果对照。乒乓球源视频为120FPS，按原脚本设定输出为30FPS慢放；网球和羽毛球保持各自原有输出策略。')

    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(MONTAGE), width=Cm(14.7))
    caption = cell.add_paragraph()
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_after = Pt(7)
    r = caption.add_run('图1 按PC参考后处理输出的RK3588结果关键帧，右上角为滑动纯推理FPS')
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
    add_text(cell, '这次测试让我认识到，板端迁移不能只看模型能否运行，还要把输入顺序、归一化方式和后处理规则一并对齐。三套模型已能在同一套RK3588运行框架下完成输出；下一步可在更长视频和真实采集画面上继续检查遮挡、快速移动和切换场景时的稳定性。')

    sign = cell.add_paragraph()
    sign.paragraph_format.space_before = Pt(8)
    sign.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = sign.add_run('填写人：李睿鑫    2026年9月27日')
    set_run(run, size=10.5)
    document.save(OUTPUT)


if __name__ == '__main__':
    main()
