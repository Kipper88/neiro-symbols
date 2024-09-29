from numpy import array, argmax, argsort
from tkinter import Frame, Canvas, Button, Label, Tk
from PIL import Image, ImageDraw
from keras.api.models import load_model

# Загрузка сохраненной модели
model = load_model('mnist_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Размер изображения для модели MNIST (28x28 пикселей)
IMAGE_SIZE = 28

class DigitRecognizer():
    def __init__(self, root):
        self.root = root
        self.root.title("Рисование цифры")
        
        # Переключение на полноэкранный режим
        self.root.attributes('-fullscreen', True)

        # Выйти из полноэкранного режима по клавише Esc
        self.root.bind('<Escape>', self.root.attributes('-fullscreen', False))

        # Создаём фрейм для размещения элементов интерфейса
        self.frame = Frame(self.root)
        self.frame.pack()

        # Холст для рисования
        self.canvas = Canvas(self.frame, width=200, height=200, bg='white')
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # Кнопка для предсказания цифры
        self.button_predict = Button(self.frame, text="Распознать", command=self.predict_digit)
        self.button_predict.grid(row=1, column=0, pady=10)

        # Кнопка для очистки холста
        self.button_clear = Button(self.frame, text="Очистить", command=self.clear_canvas)
        self.button_clear.grid(row=2, column=0, pady=10)

        # Метка для вывода результата распознавания
        self.result_label = Label(self.frame, text="Распознанная цифра: ", font=("Helvetica", 16))
        self.result_label.grid(row=0, column=1, padx=20)

        # Подготовка для рисования
        self.image = Image.new('L', (200, 200), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # Связываем события рисования мышью
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        # Рисуем на холсте и на изображении
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)
        self.draw.ellipse([x1, y1, x2, y2], fill='white')

    def clear_canvas(self):
        # Очищаем холст и изображение
        self.canvas.delete("all")
        self.image = Image.new('L', (200, 200), color=0)
        self.draw = ImageDraw.Draw(self.image)
        # Очищаем текст результата
        self.result_label.config(text="Распознанная цифра: ")

    def predict_digit(self):
        # Масштабируем и конвертируем изображение к размеру 28x28
        resized_image = self.image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = array(resized_image)

        # Нормализация изображения
        img_array = img_array / 200.0
        img_array = img_array.reshape(1, IMAGE_SIZE, IMAGE_SIZE)

        # Предсказание модели
        prediction = model.predict([img_array])[0]  # Извлекаем предсказание из массива

        # Находим два самых вероятных числа
        sorted_indices = argsort(prediction)[::-1]  # Сортируем индексы по убыванию
        top_2_indices = sorted_indices[:2]  # Выбираем два самых вероятных индекса

        top_2_probabilities = prediction[top_2_indices]

        # Формируем текст результата
        result_text = f"Распознанная цифра: {top_2_indices[0]}, вероятность {top_2_probabilities[0]:.2f}"
        if top_2_indices[0] != top_2_indices[1]:
            result_text += f"\nВозможно также: {top_2_indices[1]}, вероятность {top_2_probabilities[1]:.2f}"

        # Обновляем текст метки с результатом
        self.result_label.config(text=result_text)


# Основное приложение
root = Tk()
app = DigitRecognizer(root)
root.mainloop()
