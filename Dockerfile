# एक हल्का Python बेस इमेज चुनें
FROM python:3.10-slim-buster

# कंटेनर के अंदर आपकी ऐप की वर्किंग डायरेक्टरी सेट करें
WORKDIR /app

# requirements.txt फ़ाइल को कंटेनर में कॉपी करें
# ये पहले कॉपी होता है ताकि अगर सिर्फ़ कोड बदले और लाइब्रेरीज़ न बदलें,
# तो Docker बिल्ड करते समय कैश का इस्तेमाल कर सके, जिससे प्रोसेस तेज़ होता है।
COPY requirements.txt .

# requirements.txt में लिस्ट की गई सभी Python डिपेंडेंसी इंस्टॉल करें
# --no-cache-dir: इंस्टॉलेशन के बाद डाउनलोड की गई कैश फ़ाइलों को डिलीट कर देता है,
# जिससे फ़ाइनल Docker इमेज का साइज़ छोटा रहता है।
RUN pip install --no-cache-dir -r requirements.txt

# अब आपके प्रोजेक्ट की बाकी सभी फ़ाइलों को कंटेनर में कॉपी करें
# इसमें app.py, best_model.h5, label_encoder.pkl, और कोई भी अन्य सपोर्ट फ़ाइलें शामिल होंगी।
COPY . .

# कंटेनर के शुरू होने पर Streamlit ऐप को चलाने के लिए कमांड
# --server.port=8501: Streamlit का डिफ़ॉल्ट पोर्ट
# --server.enableCORS=true: क्रॉस-ओरिजिन रिक्वेस्ट को इनेबल करता है, जो वेब ऐप्स के लिए ज़रूरी है
# --server.enableXsrfProtection=false: XSRF प्रोटेक्शन को डिसेबल करता है (डिप्लॉयमेंट के लिए अक्सर इसकी ज़रूरत पड़ती है)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=true", "--server.enableXsrfProtection=false"]
