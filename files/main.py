import customtkinter as ctk
from customtkinter import filedialog

import json
import re
import os
import requests
from PIL import Image
import easyocr
import pdfplumber
import cv2
import numpy as np
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsNERTagger,
    Doc
)

segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)


ocr_reader = easyocr.Reader(['ru','en'], gpu=True)

""" names,surnames = [],[]

with open('datasets/fNames.txt', 'r', encoding='utf-8') as file:
    names.extend(file.read().splitlines())

with open('datasets/mNames.txt', 'r', encoding='utf-8') as file:
    names.extend(file.read().splitlines())

with open('datasets/mSurnames.txt', 'r', encoding='utf-8') as file:
    surnames.extend(file.read().splitlines()) """

GeminiAPIKEY = "AIzaSyANWdM_J5l99dr3i1wEiIikH_jN_VUSMp0"

#Gemini API функция
def ask_gemini(user_input):

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GeminiAPIKEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{"text": user_input}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.ok:
            result = response.json()
            answer = result["candidates"][0]["content"]["parts"][0]["text"]
            return answer
        else:
            return "⚠️ Ошибка API"
    except Exception as e:
        return "⚠️ Ошибка сети"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

HISTORY_FILE = "history.json"

#загрузка истории чата
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

#сохранение истории чата
def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

#функция для извлечения текста из изображения
def extract_text_from_image(file_path):
    try:
        with open(file_path, 'rb') as f:
            img_bytes = bytearray(f.read())
        img_array = np.asarray(img_bytes, dtype=np.uint8)

        # Декодирование изображения
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return "[Ошибка: Не удалось декодировать изображение]"

        results = ocr_reader.readtext(img, detail=0)

        text = " ".join(results)
        print("text", text)
        return text
    except Exception as e:
        return f"[Ошибка изображения: {e}]"
    
#функция для извлечения текста из PDF
def extract_text_from_pdf(file_path):
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        return f"[Ошибка PDF: {e}]"


#функция для скрытия личной инфы
def anonymize_text(text):
    # Номер карты: 16 цифр подряд или через пробел/дефис
    text = re.sub(r'(?:\d{4}[-\s]?){4}', '1111 2222 3333 4444', text)

    # ИИН: 12 цифр подряд
    text = re.sub(r'\b\d{12}\b', '111122223333', text)

    # Дата: гибкий формат (01.01.2024, 01/01/2024, 01-01-2024 и даже 1.1.24)
    text = re.sub(r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b', '01.01.2000', text)

    # Телефон (разные форматы с +7, 8, скобками и пробелами)
    text = re.sub(r'(\+?7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}', '+7 000 000 00 00', text)

    # ФИО: до трёх слов, с заглавных букв, возможно с отчеством
    text = re.sub(r'\b[А-ЯЁ][а-яё]+(?: [А-ЯЁ][а-яё]+){1,2}\b', 'Нуров Нур Нурович', text)

    return text

def ner_anonymize(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)

    spans = [span for span in doc.spans if span.type in {'PER', 'LOC', 'ORG', 'DATE'}]
    spans = sorted(spans, key=lambda span: span.start, reverse=True)

    for span in spans:
        if span.type == 'PER':
            replacement = 'ИМЯ ФАМИЛИЯ'
        elif span.type == 'LOC':
            replacement = 'ЛОКАЦИЯ'
        elif span.type == 'ORG':
            replacement = 'ОРГАНИЗАЦИЯ'
        elif span.type == 'DATE':
            replacement = '01.01.2000'
        else:
            continue

        text = text[:span.start] + replacement + text[span.stop:]

    return text


#класс приложения
class AnonymizerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🛡️ AnonAI")
        self.geometry("720x580")
        self.resizable(False, False)

        self.history = load_history()

        # Верхняя панель
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.description = ctk.CTkLabel(
            top_frame, 
            text="🛡️ Цензурирует личную информацию. Использует ИИ Gemini + NER",
            font=("Segoe UI", 16, "bold"),
            anchor="center"
        )
        self.description.pack(pady=5)

        # Окно чата
        self.chat_log = ctk.CTkTextbox(self, width=680, height=380, font=("Segoe UI", 14), wrap="word")
        self.chat_log.pack(padx=10, pady=5)
        self.chat_log.configure(state="disabled")

        # Нижняя панель ввода
        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(fill="both", padx=10, pady=(5, 10))

        # Поле многострочного ввода
        self.entry = ctk.CTkTextbox(bottom_frame, height=80, width=540, font=("Segoe UI", 13), wrap="word")
        self.entry.pack(side="left", padx=(0, 5), pady=5)

        # Кнопки справа
        btn_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        btn_frame.pack(side="right", padx=0)

        self.send_btn = ctk.CTkButton(btn_frame, text="⬆️ Отправить", command=self.send_message, width=100, height=40, fg_color="green")
        self.send_btn.pack(pady=(0, 5))

        self.ocr_upload = ctk.CTkButton(btn_frame, text="🖼️ Изображение/PDF", command=self.open_image, width=100, height=40)
        self.ocr_upload.pack(pady=(0, 5))

        self.clear_btn = ctk.CTkButton(btn_frame, text="🗑 Новый Чат", command=self.new_chat, width=100, height=40, fg_color="red", hover_color="#8B0000")
        self.clear_btn.pack()

        self.display_history()


    #создать новый чат (удалить старый)
    def new_chat(self):
        self.chat_log.configure(state="normal")
        self.chat_log.delete("1.0", "end")
        self.chat_log.configure(state="disabled")
        self.history = []
        save_history(self.history)
        self.entry.delete(0, "end")

    def open_image(self):
        file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[
        ("Изображения", "*.png *.jpg *.jpeg"),
        ("PDF файлы", "*.pdf")
    ])


        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            if text:
                self.entry.delete("1.0", "end")
                self.entry.insert("1.0", text)
            else:
                print("Ошибка извлечения текста из PDF")

        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            text = extract_text_from_image(file_path)
            if text:
                self.entry.delete("1.0", "end")
                self.entry.insert("1.0", text)
            else:
                print("Ошибка извлечения текста из изображения")
        else:
            print("Поддерживаются только изображения (JPG/PNG) и PDF файлы.")


    #отобразить историю
    def display_history(self):
        self.chat_log.configure(state="normal")
        for msg in self.history:
            self.chat_log.insert("end", f"👨: {msg['input']}\n🤖: {msg['output']}\n\n")
        self.chat_log.configure(state="disabled")

        self.chat_log.yview_moveto(100000)

    #отправка сообщения
    def send_message(self, event=None):
        user_input = self.entry.get("1.0", "end").strip()

        if not user_input:
            return
        
        user_input = anonymize_text(user_input)
        user_input = ner_anonymize(user_input)

        ai_input = f"Промпт: {user_input} \nИстория чата (если есть): {str(self.history)}"

        bot_response = ask_gemini(ai_input)
        self.history.append({"input": user_input, "output": bot_response})
        save_history(self.history)

        self.chat_log.configure(state="normal")
        self.chat_log.insert("end", f"👨: {user_input}\n🤖: {bot_response}\n\n")
        self.chat_log.configure(state="disabled")
        self.chat_log.see("end")
        self.entry.delete("1.0", "end")

#запуск
if __name__ == "__main__":
    app = AnonymizerApp()
    app.mainloop()
