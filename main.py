from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import gc
import json
import easyocr # <-- GOOGLE LENS GÜCÜ BURADA!

app = FastAPI(title="Akıllı Ürün Tanıma - GOOGLE LENS MODU V12")

KLASOR = "urunler"
JSON_DOSYASI = "urunler.json"

# OCR Okuyucuyu Başlat (Sadece Türkçe ve İngilizce)
# Bunu global olarak başlatıyoruz ki her seferinde yükleyip zaman kaybetmesin.
reader = easyocr.Reader(['tr', 'en'], gpu=False) 

def urun_veritabani_yukle():
    if not os.path.exists(JSON_DOSYASI): return {}
    with open(JSON_DOSYASI, "r", encoding="utf-8") as f:
        return json.load(f)

def metin_ile_bul(okunan_metinler, veritabani):
    """
    OCR ile okunan kelimeleri veritabanındaki etiketlerle karşılaştırır.
    """
    # Okunan her kelimeyi küçük harfe çevir
    okunanlar = [kelime.lower() for kelime in okunan_metinler]
    
    en_iyi_kod = None
    en_cok_eslesme = 0

    for kod, detay in veritabani.items():
        etiketler = detay.get("etiketler", [])
        
        # Kaç tane ortak kelime var?
        eslesme_sayisi = 0
        for etiket in etiketler:
            # Eğer etiket, okunan metnin içinde geçiyorsa (Örn: "BURGU" bulunduysa)
            if any(etiket in okunan for okunan in okunanlar):
                eslesme_sayisi += 1
        
        # En az 2 kelime tutmalı (Örn: Hem "Arbella" Hem "Burgu")
        # Tek kelime yetmez, çünkü "Arbella" hepsinde yazar.
        if eslesme_sayisi >= 2 and eslesme_sayisi > en_cok_eslesme:
            en_cok_eslesme = eslesme_sayisi
            en_iyi_kod = kod
            
    return en_iyi_kod, en_cok_eslesme

def goruntu_islee_ve_bul(gelen_resim_bytes):
    veritabani = urun_veritabani_yukle()
    
    # 1. GÖRÜNTÜYÜ HAZIRLA
    nparr = np.frombuffer(gelen_resim_bytes, np.uint8)
    aranan_resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if aranan_resim is None: return None, 0, "Görüntü hatası"

    # Optimize et
    yukseklik, genislik = aranan_resim.shape[:2]
    if genislik > 1000:
        oran = 1000 / genislik
        yeni_yukseklik = int(yukseklik * oran)
        aranan_resim = cv2.resize(aranan_resim, (1000, yeni_yukseklik))

    # --- YÖNTEM 1: AKAZE (ŞEKİL EŞLEŞTİRME) ---
    img_gray = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create(threshold=0.001)
    kp1, des1 = akaze.detectAndCompute(img_gray, None)
    
    en_iyi_kod = "Bulunamadı"
    en_iyi_skor = 0
    
    # ... (Buradaki AKAZE döngüsü önceki kodla aynı, kısalttım) ...
    if des1 is not None and os.path.exists(KLASOR):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        adaylar = []
        for dosya in os.listdir(KLASOR):
            if not dosya.endswith((".jpg", ".png")): continue
            try:
                db_path = os.path.join(KLASOR, dosya)
                db_img = cv2.imread(db_path, cv2.IMREAD_GRAYSCALE)
                if db_img is None: continue
                # Boyutlandırma (Hız için)
                h, w = db_img.shape[:2]
                if w > 800: # Veritabanı resimlerini de küçültelim hızlansın
                    s = 800/w
                    db_img = cv2.resize(db_img, (800, int(h*s)))

                kp2, des2 = akaze.detectAndCompute(db_img, None)
                if des2 is not None and len(des2) > 10:
                    matches = bf.match(des1, des2)
                    iyi_eslesmeler = [m for m in matches if m.distance < 60]
                    if len(iyi_eslesmeler) >= 12:
                        # Hızlıca skor hesapla (Geometriye girmeden önce sayıya bak)
                        adaylar.append((len(iyi_eslesmeler), dosya.split("_")[0].split(".")[0]))
            except: pass
        
        # Eğer güçlü bir aday varsa
        if adaylar:
            adaylar.sort(key=lambda x: x[0], reverse=True)
            if adaylar[0][0] > 20: # 20'den fazla nokta tuttuysa resimden eminiz
                en_iyi_skor = adaylar[0][0]
                en_iyi_kod = adaylar[0][1]

    # --- KARAR NOKTASI ---
    # Eğer resimden bulduysak (Skor > 30), OCR ile vakit kaybetme, hemen dön.
    if en_iyi_skor > 30:
        return en_iyi_kod, en_iyi_skor, "Görselden Bulundu"

    # --- YÖNTEM 2: OCR (YAZI OKUMA - GOOGLE LENS TAKTİĞİ) ---
    # Eğer resimden emin değilsek (Açı kötüyse, bulanıksa), yazıyı oku!
    print("Görsel yetmedi, OCR devreye giriyor...")
    
    try:
        # EasyOCR resimdeki tüm yazıları okur
        sonuclar = reader.readtext(aranan_resim, detail=0) # detail=0 sadece metinleri verir
        print(f"Okunan Yazılar: {sonuclar}")
        
        ocr_kod, ocr_eslesme = metin_ile_bul(sonuclar, veritabani)
        
        if ocr_kod:
            # OCR sonucu görselden daha güvenilirdir çünkü isim okumuştur.
            # Skoru yapay olarak 100 yapıyoruz ki sistem kabul etsin.
            return ocr_kod, 100, f"Yazıdan Bulundu ({ocr_eslesme} kelime)"
            
    except Exception as e:
        print(f"OCR Hatası: {e}")

    # Hiçbiri bulamadıysa
    return en_iyi_kod, en_iyi_skor, "Görsel Skoru"

@app.get("/")
def ana_sayfa():
    return {"durum": "aktif", "mod": "V12 - HIBRIT (Görsel + Yazı Okuma)"}

@app.post("/tara")
async def urun_tara(file: UploadFile = File(...)):
    resim_verisi = await file.read()
    kod, skor, kaynak = goruntu_islee_ve_bul(resim_verisi)
    
    detaylar = urun_bilgisi_getir(kod) # Bu fonksiyonu V11 kodundaki gibi yukarıya eklemelisin
    
    # Baraj
    LIMIT = 15
    
    if skor >= LIMIT and kod != "Bulunamadı":
        return {
            "sonuc": True, 
            "urun_kodu": kod, 
            "guven_skoru": skor, 
            "kaynak": kaynak,
            "mesaj": "Bulundu",
            "urun_detay": detaylar
        }
    else:
        return {"sonuc": False, "mesaj": "Eşleşme Yok"}

# Yardımcı fonksiyon (V11'den kopyalanacak)
def urun_bilgisi_getir(kod):
    if not os.path.exists(JSON_DOSYASI): return {"ad": kod, "fiyat": "?", "etiketler": []}
    with open(JSON_DOSYASI, "r", encoding="utf-8") as f:
        veri = json.load(f)
    return veri.get(kod, {"ad": kod, "fiyat": "?", "etiketler": []})
