import sys
from PyQt5 import QtWidgets, QtCore
import Proj_ui
import main
import threading
import traceback


class AppStart(QtWidgets.QMainWindow, Proj_ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)

        self.pushButton.clicked.connect(self.start_process)

    def start_process(self):
        self.pushButton.setEnabled(False)
        self.progressBar.setGeometry(QtCore.QRect(360, 570, 581, 31))
        self.progressBar.setProperty("value", 20)
        self.progressBar.setObjectName("progressBar")
        self.label_9.setText("Собираю датасет")
        # tread = threading.Thread(target=self.start_thread)
        # tread.daemon = True
        # tread.start()
        self.start_thread()
        self.pushButton.setEnabled(True)

    def start_thread(self):
        try:
            colors = 1 if self.comboBox_4.currentText() == "Чёрно-белая" else 3
            response = main.create_model(self.textEdit_13.toPlainText(), self.comboBox_2.currentText(), colors,
                                         self.textEdit_2.toPlainText(), self.comboBox_3.currentText(),
                                         self.textEdit_4.toPlainText(), self.comboBox.currentText(),
                                         self.textEdit_5.toPlainText(), '20',
                                         self.progressBar, self.label_9, self.label_10)
            if response == "Error_Model_Not_Exist":
                self.label_9.setText("Модель не найдена!")
            elif response == "Error_Need_Time_Or_Epochs":
                self.label_9.setText("Нужны данные эпох или времени!")
            elif response == "Error_Dataset_Dir":
                self.label_9.setText("Датасет не найден!")
        except:
            traceback.print_exc()


def start():
    app = QtWidgets.QApplication(sys.argv)
    window = AppStart()
    window.show()
    app.exec_()


if __name__ == '__main__':
    start()
