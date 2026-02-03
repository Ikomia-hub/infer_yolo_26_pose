"""
Main Ikomia plugin module.
Ikomia Studio and Ikomia API use it to load algorithms dynamically.
"""
from ikomia import dataprocess


class IkomiaPlugin(dataprocess.CPluginProcessInterface):
    """
    Interface class to integrate the process with Ikomia application.
    Inherits PyDataProcess.CPluginProcessInterface from Ikomia API.
    """

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        """Instantiate process object."""
        from infer_yolo_26_pose.infer_yolo_26_pose_process import InferYolo26PoseFactory
        return InferYolo26PoseFactory()

    def get_widget_factory(self):
        """Instantiate associated widget object."""
        from infer_yolo_26_pose.infer_yolo_26_pose_widget import InferYolo26PoseWidgetFactory
        return InferYolo26PoseWidgetFactory()
