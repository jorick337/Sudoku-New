# Sudoku

<p align="center">
   <img src="https://github.com/jorick337/Sudoku/blob/main/MinorFiles/Images(my)/Gameplay.gif">
</p>

<p align="center">
   <img src="https://img.shields.io/badge/Engine-Unity%206000.0.38f1-blueviolet?style=&logo=unity" alt="Engine">
   <img src="https://img.shields.io/badge/Platform-Windows, Linux, MacOs %20-brightgreen?style=&logo=android" alt="Platform">
   <img src="https://img.shields.io/badge/Version-1.0.1-blue" alt="Game Version">
   <img src="https://img.shields.io/badge/Release Date-17.02.2025-red" alt="Release Date">
</p>

## Об игре

**Sudoku** это классическое приложение-судоку с добавлением искусственного интеллекта, который дает новый игровой опыт игроку. Игра предназначена для обучения.

В проекте использовался PyTorch для генерации модели нейросети и ONNX для экспорта модели в такой формат, чтобы использовать в пакете от Unity Sentis для использования нейросети(/MinorFiles/NeuroModel/...)

## О проекте

* **Дата завершения:** Февраль 1, 2025
* **Последнее обновление:** Февраль 17, 2025
* Проект был сделан на **[Unity Engine](https://unity.com/)**
* **Версия движка:** 6000.0.38f1

## Использованные инструменты:

1. **PyTorch** - фреймворк машинного обучения для языка Python с открытым исходным кодом, созданный на базе Torch.
2. **ONNX** - библиотека Python, которая преобразует модель в формат ONNX.
2. **Sentis** - пакет из PacketManager Unity, который использует ONNX обученные нейро модели для их реализации.
