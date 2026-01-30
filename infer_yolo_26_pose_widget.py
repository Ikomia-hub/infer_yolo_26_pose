from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_yolo_26_pose.infer_yolo_26_pose_process import InferYolo26PoseParam

from PyQt5.QtWidgets import *
from torch.cuda import is_available


class InferYolo26PoseWidget(core.CWorkflowTaskWidget):
    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferYolo26PoseParam()
        else:
            self.parameters = param

        self.grid_layout = QGridLayout()

        self.check_cuda = pyqtutils.append_check(
            self.grid_layout, "Cuda", self.parameters.cuda and is_available())
        self.check_cuda.setEnabled(is_available())

        self.combo_model = pyqtutils.append_combo(self.grid_layout, "Model name")
        self.combo_model.addItem("yolo26n-pose")
        self.combo_model.addItem("yolo26s-pose")
        self.combo_model.addItem("yolo26m-pose")
        self.combo_model.addItem("yolo26l-pose")
        self.combo_model.addItem("yolo26x-pose")
        self.combo_model.setCurrentText(self.parameters.model_name)

        self.spin_input_size = pyqtutils.append_spin(
            self.grid_layout,
            "Input size",
            self.parameters.input_size
        )

        self.spin_conf_thres = pyqtutils.append_double_spin(
            self.grid_layout,
            "Confidence threshold",
            self.parameters.conf_thres,
            min=0.,
            max=1.,
            step=0.01,
            decimals=2
        )

        self.spin_iou_thres = pyqtutils.append_double_spin(
            self.grid_layout,
            "Confidence IoU",
            self.parameters.iou_thres,
            min=0.,
            max=1.,
            step=0.01,
            decimals=2
        )

        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)
        self.set_layout(layout_ptr)

    def on_apply(self):
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.input_size = self.spin_input_size.value()
        self.parameters.conf_thres = self.spin_conf_thres.value()
        self.parameters.iou_thres = self.spin_iou_thres.value()
        self.parameters.update = True

        self.emit_apply(self.parameters)


class InferYolo26PoseWidgetFactory(dataprocess.CWidgetFactory):
    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        self.name = "infer_yolo_26_pose"

    def create(self, param):
        return InferYolo26PoseWidget(param, None)
