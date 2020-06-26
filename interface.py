import sys
from PyQt5 import QtWidgets
import Proj_ui
import main
import traceback
import gif
import asyncio


class AppStart(QtWidgets.QMainWindow, Proj_ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.pushButton.clicked.connect(self.start_process)
        self.pushButton_1.clicked.connect(self.generate_image)
        self.pushButton_2.clicked.connect(self.generate_gif)

    def start_process(self):
        self.pushButton.setEnabled(False)
        self.start_thread()
        self.pushButton.setEnabled(True)

    def start_thread(self):
        try:
            self.label_9.setText("Собираю датасет")
            colors = 1 if self.comboBox_4.currentText() == "Чёрно-белая" else 3
            response = asyncio.get_event_loop()
            response.run_until_complete(main.create_model(self.textEdit_13.toPlainText(), self.comboBox_2.currentText(), colors,
                                         self.textEdit_2.toPlainText(), self.comboBox_3.currentText(),
                                         self.textEdit_4.toPlainText(), self.comboBox.currentText(),
                                         self.textEdit_5.toPlainText(), '20',
                                         self.label_9, self.textEdit_10.toPlainText(), self.progressBar))
            response.close()
            if response == "Error_Model_Not_Exist":
                self.label_9.setText("Модель не найдена!")
            elif response == "Error_Need_Time_Or_Epochs":
                self.label_9.setText("Нужны данные эпох или времени!")
            elif response == "Error_Dataset_Dir":
                self.label_9.setText("Датасет не найден!")
        except:
            traceback.print_exc()
            self.label_9.setText("Произошла ошибка")

    def generate_image(self):
        if self.textEdit_5.toPlainText() == "":
            self.label_9.setText("Укажите используемую модель в \"Путь к файлу модели\"!")
        elif self.textEdit_6.toPlainText() == "":
            self.label_9.setText("Укажите название генерируемого файла!")
        else:
            try:
                main.create(self.textEdit_5.toPlainText(), self.textEdit_6.toPlainText())
            except:
                traceback.print_exc()
                self.label_9.setText("Произошла ошибка")

    def generate_gif(self):
        if self.textEdit_6.toPlainText() == "":
            self.label_9.setText("Укажите название генерируемого файла!")
        else:
            try:
                gif.gif_gen()
            except:
                traceback.print_exc()
                self.label_9.setText("Произошла ошибка")


def start():
    app = QtWidgets.QApplication(sys.argv)
    window = AppStart()
    window.show()
    app.exec_()


if __name__ == '__main__':
    start()
