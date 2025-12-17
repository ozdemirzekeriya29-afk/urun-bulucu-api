from fastapi import FastAPI, File, UploadFile, Form
import cv2
import numpy as np
import os
import json
import easyocr
from difflib import SequenceMatcher

app = FastAPI(title="Akıllı Ürün Bulucu v2")

KLASOR = "urunler"
JSON_DOSYASI = "urunler.json"
reader = easyocr.Reader(['tr', 'en'], gpu=False) 

# --- YARDIMCI FONKSİYONLAR ---
def veritabani_yukle():
    if not os.path.exists(JSON_DOSYASI): return {}
    with open(JSON_DOSYASI, "r", encoding="utf-8") as f:
        return json.load(f)

# --- ANA ANALİZ (RESİM) ---
def analiz_et(gelen_resim_bytes):
    veritabani = veritabani_yukle()
    nparr = np.frombuffer(gelen_resim_bytes, np.uint8)
    resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. OCR (Yazı Okuma)
    try:
        okunanlar = reader.readtext(resim, detail=0)
        okunan_kelimeler = [w.lower() for w in okunanlar]
    except:
        okunan_kelimeler = []

    en_iyi_skor = 0
    en_iyi_urun = None
    bulunan_kod = ""

    # 2. Kelime Eşleştirme
    for kod, detay in veritabani.items():
        puan = 0
        anahtar_kelimeler = detay.get("anahtar_kelimeler", [])
        
        # Resimde bu anahtar kelimeler geçiyor mu?
        for anahtar in anahtar_kelimeler:
            for okunan in okunan_kelimeler:
                if anahtar in okunan:
                    puan += 1
        
        if puan > en_iyi_skor:
            en_iyi_skor = puan
            en_iyi_urun = detay
            bulunan_kod = kod

    # Basit bir baraj puanı (En az 1 kelime tutmalı)
    if en_iyi_skor >= 1:
        return bulunan_kod, en_iyi_skor * 20, en_iyi_urun
    else:
        return "Bulunamadı", 0, {}

# --- YENİ ÖZELLİK: METİN İLE ARAMA ---
@app.post("/ara_metin")
async def metinle_ara(arama: str = Form(...)):
    print(f"Gelen Arama İsteği: {arama}")
    veritabani = veritabani_yukle()
    arama = arama.lower()
    
    en_iyi_eslesme = None
    en_yuksek_oran = 0
    bulunan_kod = ""

    # Tüm ürünlerin anahtar kelimelerine bak
    for kod, detay in veritabani.items():
        anahtar_kelimeler = detay.get("anahtar_kelimeler", [])
        
        # Kullanıcının yazdığı kelime, anahtar kelimelerin içinde geçiyor mu?
        eslesme_sayisi = 0
        for kelime in anahtar_kelimeler:
            if kelime in arama or arama in kelime:
                eslesme_sayisi += 1
        
        if eslesme_sayisi > en_yuksek_oran:
            en_yuksek_oran = eslesme_sayisi
            en_iyi_eslesme = detay
            bulunan_kod = kod

    if en_iyi_eslesme and en_yuksek_oran > 0:
        return {
            "sonuc": True,
            "urun_kodu": bulunan_kod,
            "urun_detay": en_iyi_eslesme,
            "mesaj": "Kelime eşleşmesi bulundu"
        }
    else:
        return {"sonuc": False, "mesaj": "Benzer ürün bulunamadı"}

# --- ESKİ RESİM TARAMA ---
@app.post("/tara")
async def urun_tara(file: UploadFile = File(...)):
    resim_verisi = await file.read()
    kod, skor, detay = analiz_et(resim_verisi)
    
    if kod != "Bulunamadı":
        return {"sonuc": True, "urun_kodu": kod, "guven_skoru": skor, "urun_detay": detay}
    else:
        return {"sonuc": False, "mesaj": "Resimden anlaşılamadı"}
