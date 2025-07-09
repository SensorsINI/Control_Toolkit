from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QGroupBox,
    QRadioButton,
    QDialogButtonBox,
)
from PyQt6.QtCore import Qt

from Control_Toolkit.others.globals_and_utils import (
    get_available_controller_names,
    get_available_optimizer_names,
    get_controller_name,
    get_optimizer_name,
)


class SelectionDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Controller & Optimizer")
        self.resize(400, 300)

        layout = QVBoxLayout(self)

        # Controllers group
        ctrl_names = get_available_controller_names()
        box_ctrl = QGroupBox("Controllers")
        vbox_ctrl = QVBoxLayout()
        self.rbs_controllers = []
        for name in ctrl_names:
            rb = QRadioButton(name)
            vbox_ctrl.addWidget(rb)
            self.rbs_controllers.append(rb)
        if self.rbs_controllers:
            self.rbs_controllers[0].setChecked(True)
        box_ctrl.setLayout(vbox_ctrl)
        layout.addWidget(box_ctrl)

        # Optimizers group
        opt_names = get_available_optimizer_names()
        box_opt = QGroupBox("Optimizers")
        vbox_opt = QVBoxLayout()
        self.rbs_optimizers = []
        for name in opt_names:
            rb = QRadioButton(name)
            vbox_opt.addWidget(rb)
            self.rbs_optimizers.append(rb)
        if self.rbs_optimizers:
            self.rbs_optimizers[0].setChecked(True)
        box_opt.setLayout(vbox_opt)
        layout.addWidget(box_opt)

        # OK / Cancel buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_selection(self):
        """
        Returns:
            Tuple[str, str]: (controller_name, optimizer_name)
        """
        ctrl = None
        for idx, rb in enumerate(self.rbs_controllers):
            if rb.isChecked():
                ctrl, _ = get_controller_name(controller_idx=idx)
                break
        opt = None
        for idx, rb in enumerate(self.rbs_optimizers):
            if rb.isChecked():
                opt, _ = get_optimizer_name(optimizer_idx=idx)
                break
        return ctrl, opt


def choose_controller_and_optimizer():
    import sys
    app = QApplication(sys.argv)
    dlg = SelectionDialog()
    if dlg.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)
    return dlg.get_selection()
