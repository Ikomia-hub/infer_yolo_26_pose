"""
Main Ikomia plugin module.
Ikomia Studio and Ikomia API use it to load algorithms dynamically.
"""
from ikomia import dataprocess
from infer_yolo_26_pose.infer_yolo_26_pose_process import InferYolo26PoseFactory
from infer_yolo_26_pose.infer_yolo_26_pose_process import InferYolo26PoseParamFactory


class IkomiaPlugin(dataprocess.CPluginProcessInterface):
    """
    Interface class to integrate the process with Ikomia application.
    Inherits PyDataProcess.CPluginProcessInterface from Ikomia API.
    """
    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        """Instantiate process object."""
        return InferYolo26PoseFactory()

    def get_widget_factory(self):
        """Instantiate associated widget object."""
        from infer_yolo_26_pose.infer_yolo_26_pose_widget import InferYolo26PoseWidgetFactory
        return InferYolo26PoseWidgetFactory()

    def get_param_factory(self):
        """Instantiate algorithm parameters object."""
        return InferYolo26PoseParamFactory()
