# AI High-res SWIR Multispectral SPC

## Описание проекта
Проект посвящен разработке мультиспектральной инфракрасной однопиксельной камеры с применением методов глубокого обучения. Камера предназначена для визуализации объектов в SWIR-диапазоне и анализа их спектральных характеристик. Основное применение – анализ материалв на конвейерных лентах, включая обнаружение дефектов на продуктах (контроль качества в пищевом секторе) и исследование кернов (горных образцов) для анализа их состава, что имеет важное значение в геологических и горнодобывающих задачах. В проекте применяются методы сегментации к редкому типу данных, полученных с самостоятельно полностью собранной мультиспектральной инфракрасной однопиксельной камеры, ранее не использовавшихся в подобных задачах. Особое внимание уделено исследованию влияния различных функций ошибки на эффективность сегментации, что является важным вкладом в область компьютерного зрения, особенно при работе с изображениями, полученными в инфракрасном диапазоне с низким разрешением. Применение адаптивных функций ошибки и трансферного обучения позволило добиться значительных улучшений в сегментации, что подтверждается оценками метрик качества.

## Основные компоненты
- **Однопиксельная SWIR-камера** с источниками на 800 нм, 1050 нм и 1550 нм.
- **Обработка данных детектора** с методом наложения паттернов освещения.
- **Глубокое обучение** для повышения разрешения и сегментации объектов.

## Ход выполнения
### Оптическая часть
- Разработана и собрана оптическая установка для однопиксельной визуализации.
- Проведено моделирование визуализации объектов в ИК-диапазоне.
- Создана модель визуализации кернов на конвейере.

### AI-часть
- Собраны датасеты на сегментацию дефектов продуктов и минералов для получения претрейнов:
  - 4060 изображения для продуктов из 14 датасетов.
  - 15000 синтетических изображений для минералов.
- Далее на этих данных были обучены различные архитектуры сегментации (претрейны)  DeepLabv3 (ResNet50 и Resnet101), LRASPP и U-Net (encoder - ResNet 50 backbone) и оценено качество их работы, что стало важной отправной точкой для последующего дообучения полученных весов на реальных инфракрасных изображениях продуктов и минералов. Были использованы следующие архитектуры:
  - DeepLabv3 (ResNet50)
  - DeepLabv3 (Resnet101)
  - LRASPP (MobileNetV3-Large backbone)
  - U-Net (encoder - ResNet 50 backbone)
- Проведено дообучение моделей Segment Anything (SAM-2.1) для будущей разметки собственного датасета, полученного с собранной установки.
- Собраны и размечены собственные наборы данных по 1000 инфракрасных изображений для продуктов и минералов.
- Особое внимание было уделено формализации и разработке функций ошибки, способных эффективно учитывать особенности задач сегментации тонких и слабо выраженных неоднородностей.
- Далее было проведено трансферное обучение на собранных данных, а также выполнена серия экспериментов по исследованию влияния различных функций ошибки на значение целевых метрик качества выбранных архитектур сегментации.Результаты показали, что использование предложенной комбинированной функции ошибки в сочетании с адаптивными параметрами позволяет улучшить метрики качества сегментации, IoU на 8.3%, Dice на 7.0%, mAP на 7.9% в случае сегментации изображений продуктов с неоднородностями, а для минералов - IoU на 7.1%, Dice на 5.2%, mAP на 8.7%.
- Также было проведено повышение качества изображений путём подавления шума (с помощью U-Net подобной архитектуры) и повышения разрешения на основе super-resolution (использовались архитектуры SMFANet и MobileSR).

## Используемые технологии
- **Глубокое обучение** (Segment Anything, DeepLabV3, U-Net, LRASPP, SMFANet, MobileSR, Transfer Learning).
- **Моделирование визуализации объектов с помощью MatLab**.
- **Orange PI** для работы с инференсом моделей (планируется во 2 этапе НИРСИИ).


## Будущие планы
- Расширение набора реальных данных для улучшения работы моделей.
- Ускорение получение кадров с установки.
- Использование в производственном контроле качества продукции.
- Публикации и доклады на научных конференциях.
  
---
Проект выполняется в рамках НИРСИИ  (Университета ИТМО).
