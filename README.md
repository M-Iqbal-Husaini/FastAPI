# FastAPI Project

Backend API sederhana yang dibangun menggunakan **FastAPI**.
Project ini bertujuan untuk menyediakan layanan API yang cepat, ringan, dan mudah dikembangkan menggunakan Python.

FastAPI merupakan framework modern untuk membangun API berbasis HTTP di Python yang mendukung **type hints, validasi data otomatis, serta dokumentasi API otomatis menggunakan OpenAPI (Swagger UI)**.

---

# 🚀 Features

* REST API menggunakan **FastAPI**
* Dokumentasi API otomatis (Swagger UI & ReDoc)
* Validasi request menggunakan **Pydantic**
* Struktur project sederhana dan mudah dikembangkan
* Performa tinggi dengan **ASGI server (Uvicorn)**

---

# 🧱 Tech Stack

* **Python**
* **FastAPI**
* **Pydantic**
* **Uvicorn**

---

# 📦 Installation

Ikuti langkah berikut untuk menjalankan project secara lokal.

## 1 Clone Repository

```bash
git clone https://github.com/M-Iqbal-Husaini/FastAPI.git
```

## 2 Masuk ke Folder Project

```bash
cd FastAPI
```

## 3 Buat Virtual Environment

```bash
python -m venv venv
```

Aktifkan environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

---

## 4 Install Dependency

Jika terdapat `requirements.txt`

```bash
pip install -r requirements.txt
```

Jika belum ada:

```bash
pip install fastapi uvicorn
```

---

# ▶️ Menjalankan Server

Jalankan aplikasi dengan:

```bash
uvicorn main:app --reload
```

Server akan berjalan di:

```
http://127.0.0.1:8000
```

---

# 📚 Dokumentasi API

FastAPI menyediakan dokumentasi otomatis.

Swagger UI:

```
http://127.0.0.1:8000/docs
```

ReDoc:

```
http://127.0.0.1:8000/redoc
```

Di halaman ini kamu bisa:

* melihat endpoint API
* mencoba request langsung dari browser
* melihat struktur request dan response

---

# 📂 Struktur Project

Contoh struktur sederhana:

```
FastAPI
│
├── main.py
├── requirements.txt
├── routers
│   └── api.py
├── models
│   └── schema.py
└── services
```

Penjelasan:

* **main.py** → entry point aplikasi
* **routers** → endpoint API
* **models** → schema data
* **services** → business logic

---

# 🧪 Contoh Endpoint

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI"}
```

Endpoint dapat diakses di:

```
GET /
```

Response:

```
{
  "message": "Hello FastAPI"
}
```

---

# 🎯 Tujuan Project

Project ini dibuat untuk:

* mempelajari framework **FastAPI**
* membangun backend API menggunakan Python
* memahami pembuatan REST API modern

---

# 👨‍💻 Author

**Muhammad Iqbal Husaini**

---

# 📄 License

Project ini bersifat **open source** dan dapat digunakan untuk tujuan pembelajaran maupun pengembangan lebih lanjut.

